import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, classification_report
import tensorflow as tf

class UncertaintyEstimator:
    def __init__(self, model, n_samples=20):
        self.model = model
        self.n_samples = n_samples
    
    def predict_with_uncertainty(self, image_batch):
        """Predict with uncertainty using Monte Carlo Dropout"""
        predictions = []
        
        for _ in range(self.n_samples):
            # Enable dropout during inference
            pred = self.model(image_batch, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty
    
    def analyze_uncertainty(self, test_df, disease_labels, data_generator, num_samples=50):
        """Analyze uncertainty on test samples"""
        print("Analyzing prediction uncertainty...")
        
        all_predictions = []
        all_uncertainties = []
        all_true_labels = []
        
        sample_df = test_df.sample(n=min(num_samples, len(test_df)))
        
        for _, row in sample_df.iterrows():
            img_path = os.path.join(data_generator.images_dir, row['Image Index'])
            img = data_generator.load_and_preprocess_image(img_path)
            
            if img is not None:
                img_normalized = img / 255.0
                img_batch = np.expand_dims(img_normalized, axis=0)
                
                mean_pred, uncertainty = self.predict_with_uncertainty(img_batch)
                
                all_predictions.append(mean_pred[0])
                all_uncertainties.append(uncertainty[0])
                all_true_labels.append(row[disease_labels].values)
        
        return np.array(all_predictions), np.array(all_uncertainties), np.array(all_true_labels)
    
    def plot_uncertainty_analysis(self, predictions, uncertainties, true_labels, disease_labels):
        """Plot uncertainty analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Uncertainty Analysis', fontsize=16)
        
        # Uncertainty distribution
        avg_uncertainty = np.mean(uncertainties, axis=1)
        axes[0, 0].hist(avg_uncertainty, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Average Uncertainty')
        axes[0, 0].set_xlabel('Average Uncertainty')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Uncertainty vs Confidence
        max_confidence = np.max(predictions, axis=1)
        axes[0, 1].scatter(max_confidence, avg_uncertainty, alpha=0.6, color='coral')
        axes[0, 1].set_title('Uncertainty vs Confidence')
        axes[0, 1].set_xlabel('Max Prediction Confidence')
        axes[0, 1].set_ylabel('Average Uncertainty')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Disease-specific uncertainty
        disease_uncertainties = np.mean(uncertainties, axis=0)
        y_pos = np.arange(len(disease_labels))
        axes[1, 0].barh(y_pos, disease_uncertainties, color='lightgreen', edgecolor='black')
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(disease_labels, fontsize=8)
        axes[1, 0].set_title('Average Uncertainty by Disease')
        axes[1, 0].set_xlabel('Average Uncertainty')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Calibration plot
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies = []
        confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (max_confidence > bin_lower) & (max_confidence <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = ((predictions[in_bin] > 0.5) == (true_labels[in_bin] > 0.5)).mean()
                avg_confidence_in_bin = max_confidence[in_bin].mean()
                accuracies.append(accuracy_in_bin)
                confidences.append(avg_confidence_in_bin)
        
        axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        axes[1, 1].plot(confidences, accuracies, 'o-', color='red', label='Model Calibration')
        axes[1, 1].set_title('Calibration Plot')
        axes[1, 1].set_xlabel('Confidence')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class GradCAMVisualizer:
    def __init__(self, model, layer_name='conv5_block3_out'):
        self.model = model
        self.layer_name = layer_name
        
    def generate_gradcam(self, image, class_index):
        """Generate Grad-CAM heatmap"""
        try:
            # Get the ResNet50 base model
            base_model = None
            for layer in self.model.layers:
                if hasattr(layer, 'layers'):  # This is the ResNet50 base model
                    base_model = layer
                    break
            
            if base_model is None:
                return np.zeros((224, 224))
            
            # Create grad model
            grad_model = tf.keras.models.Model(
                [self.model.inputs],
                [base_model.get_layer(self.layer_name).output, self.model.output]
            )
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image)
                loss = predictions[:, class_index]
            
            # Get gradients
            grads = tape.gradient(loss, conv_outputs)
            
            # Pool gradients over spatial dimensions
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight feature maps by gradients
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # Normalize heatmap
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            # Resize to original image size
            heatmap = tf.image.resize(heatmap[..., tf.newaxis], (224, 224))
            heatmap = tf.squeeze(heatmap)
            
            return heatmap.numpy()
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            return np.zeros((224, 224))
    
    def visualize_gradcam(self, image_path, disease_labels, data_generator, top_k=3):
        """Visualize Grad-CAM for top predicted diseases"""
        # Load and preprocess image
        img = data_generator.load_and_preprocess_image(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return None, None
        
        img_normalized = img / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Get predictions
        predictions = self.model.predict(img_batch, verbose=0)[0]
        
        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        # Create visualization
        fig, axes = plt.subplots(2, top_k, figsize=(15, 8))
        if top_k == 1:
            axes = axes.reshape(2, 1)
        
        for i, class_idx in enumerate(top_indices):
            disease_name = disease_labels[class_idx]
            confidence = predictions[class_idx]
            
            # Generate Grad-CAM
            heatmap = self.generate_gradcam(img_batch, class_idx)
            
            # Original image
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f'{disease_name}\nConfidence: {confidence:.3f}')
            axes[0, i].axis('off')
            
            # Grad-CAM overlay
            axes[1, i].imshow(img, cmap='gray')
            axes[1, i].imshow(heatmap, cmap='jet', alpha=0.4)
            axes[1, i].set_title('Grad-CAM Heatmap')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return predictions, top_indices

class ModelEvaluator:
    def __init__(self, model, data_generator, disease_labels):
        self.model = model
        self.data_generator = data_generator
        self.disease_labels = disease_labels
    
    def evaluate_model(self, test_df, num_samples=100):
        """Comprehensive model evaluation"""
        print("=== Model Evaluation ===")
        
        # Sample test data for faster evaluation
        test_sample = test_df.sample(n=min(num_samples, len(test_df)))
        
        all_predictions = []
        all_true_labels = []
        
        for _, row in test_sample.iterrows():
            img_path = os.path.join(self.data_generator.images_dir, row['Image Index'])
            img = self.data_generator.load_and_preprocess_image(img_path)
            
            if img is not None:
                img_normalized = img / 255.0
                img_batch = np.expand_dims(img_normalized, axis=0)
                
                pred = self.model.predict(img_batch, verbose=0)[0]
                all_predictions.append(pred)
                all_true_labels.append(row[self.disease_labels].values)
        
        predictions = np.array(all_predictions)
        true_labels = np.array(all_true_labels)
        
        return self.calculate_metrics(predictions, true_labels)
    
    def calculate_metrics(self, predictions, true_labels):
        """Calculate various evaluation metrics"""
        results = {}
        
        # AUC for each disease
        disease_aucs = {}
        for i, disease in enumerate(self.disease_labels):
            if len(np.unique(true_labels[:, i])) > 1:
                try:
                    auc = roc_auc_score(true_labels[:, i], predictions[:, i])
                    disease_aucs[disease] = auc
                except:
                    disease_aucs[disease] = 0.5  # Random performance if calculation fails
            else:
                disease_aucs[disease] = 0.5
        
        # Overall metrics
        results['disease_aucs'] = disease_aucs
        results['mean_auc'] = np.mean(list(disease_aucs.values()))
        results['predictions'] = predictions
        results['true_labels'] = true_labels
        
        print(f"Mean AUC: {results['mean_auc']:.4f}")
        print("\nDisease-specific AUCs:")
        for disease, auc in sorted(disease_aucs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {disease}: {auc:.4f}")
        
        return results
    
    def plot_evaluation_results(self, results):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Evaluation Results', fontsize=16)
        
        # AUC scores by disease
        disease_aucs = results['disease_aucs']
        diseases = list(disease_aucs.keys())
        aucs = list(disease_aucs.values())
        
        y_pos = np.arange(len(diseases))
        bars = axes[0, 0].barh(y_pos, aucs, color='skyblue', edgecolor='navy')
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels(diseases, fontsize=8)
        axes[0, 0].set_xlabel('AUC Score')
        axes[0, 0].set_title('AUC Scores by Disease')
        axes[0, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Random')
        axes[0, 0].axvline(x=results['mean_auc'], color='green', linestyle='-', alpha=0.7, label='Mean AUC')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[0, 0].text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        # Prediction distribution
        predictions = results['predictions']
        axes[0, 1].hist(predictions.flatten(), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Prediction Score Distribution')
        axes[0, 1].set_xlabel('Prediction Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Threshold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Confusion matrix for binary predictions (using 0.5 threshold)
        binary_preds = (predictions > 0.5).astype(int)
        true_labels = results['true_labels']
        
        # Calculate overall accuracy metrics
        tp = np.sum((binary_preds == 1) & (true_labels == 1))
        tn = np.sum((binary_preds == 0) & (true_labels == 0))
        fp = np.sum((binary_preds == 1) & (true_labels == 0))
        fn = np.sum((binary_preds == 0) & (true_labels == 1))
        
        cm = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   ax=axes[1, 0])
        axes[1, 0].set_title('Overall Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Performance summary
        axes[1, 1].axis('off')
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        summary_text = f"""Performance Summary:
        
Mean AUC: {results['mean_auc']:.4f}
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-Score: {f1:.4f}

Total Samples: {len(predictions)}
Total Predictions: {len(predictions) * len(self.disease_labels)}

Best Performing Diseases:
"""
        
        # Add top 3 diseases by AUC
        sorted_diseases = sorted(disease_aucs.items(), key=lambda x: x[1], reverse=True)
        for i, (disease, auc) in enumerate(sorted_diseases[:3]):
            summary_text += f"{i+1}. {disease}: {auc:.4f}\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        plt.show()

class FairnessAnalyzer:
    def __init__(self, model, data_generator):
        self.model = model
        self.data_generator = data_generator
    
    def evaluate_by_group(self, test_df, group_column, disease_labels, num_samples=50):
        """Evaluate model performance by demographic groups"""
        print(f"=== Fairness Analysis by {group_column} ===")
        
        groups = test_df[group_column].unique()
        group_results = {}
        
        for group in groups:
            group_df = test_df[test_df[group_column] == group].sample(
                n=min(num_samples, len(test_df[test_df[group_column] == group]))
            )
            
            predictions = []
            true_labels = []
            
            for _, row in group_df.iterrows():
                img_path = os.path.join(self.data_generator.images_dir, row['Image Index'])
                img = self.data_generator.load_and_preprocess_image(img_path)
                
                if img is not None:
                    img_normalized = img / 255.0
                    img_batch = np.expand_dims(img_normalized, axis=0)
                    
                    pred = self.model.predict(img_batch, verbose=0)[0]
                    predictions.append(pred)
                    true_labels.append(row[disease_labels].values)
            
            if predictions:
                predictions = np.array(predictions)
                true_labels = np.array(true_labels)
                
                # Calculate AUCs for this group
                group_aucs = {}
                for i, disease in enumerate(disease_labels):
                    if len(np.unique(true_labels[:, i])) > 1:
                        try:
                            auc = roc_auc_score(true_labels[:, i], predictions[:, i])
                            group_aucs[disease] = auc
                        except:
                            group_aucs[disease] = 0.5
                    else:
                        group_aucs[disease] = 0.5
                
                group_results[group] = {
                    'aucs': group_aucs,
                    'mean_auc': np.mean(list(group_aucs.values())),
                    'sample_size': len(predictions)
                }
                
                print(f"{group}: Mean AUC = {group_results[group]['mean_auc']:.4f} (n={len(predictions)})")
        
        return group_results
    
    def calculate_fairness_metrics(self, group_results):
        """Calculate fairness metrics"""
        groups = list(group_results.keys())
        if len(groups) < 2:
            return {}
        
        # Calculate demographic parity and equalized odds
        mean_aucs = [group_results[group]['mean_auc'] for group in groups]
        
        fairness_metrics = {
            'auc_difference': max(mean_aucs) - min(mean_aucs),
            'auc_ratio': min(mean_aucs) / max(mean_aucs) if max(mean_aucs) > 0 else 0,
            'group_aucs': {group: group_results[group]['mean_auc'] for group in groups}
        }
        
        print(f"\nFairness Metrics:")
        print(f"AUC Difference: {fairness_metrics['auc_difference']:.4f}")
        print(f"AUC Ratio: {fairness_metrics['auc_ratio']:.4f}")
        
        return fairness_metrics
    
    def plot_fairness_analysis(self, group_results, fairness_metrics):
        """Plot fairness analysis results"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Fairness Analysis Results', fontsize=16)
        
        # Group performance comparison
        groups = list(group_results.keys())
        mean_aucs = [group_results[group]['mean_auc'] for group in groups]
        sample_sizes = [group_results[group]['sample_size'] for group in groups]
        
        bars = axes[0].bar(groups, mean_aucs, color=['skyblue', 'lightcoral'][:len(groups)], 
                          edgecolor='navy', alpha=0.7)
        axes[0].set_title('Mean AUC by Group')
        axes[0].set_ylabel('Mean AUC')
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)
        
        # Add sample size labels
        for i, (bar, size) in enumerate(zip(bars, sample_sizes)):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'n={size}', ha='center', va='bottom', fontsize=10)
            axes[0].text(bar.get_x() + bar.get_width()/2., height/2,
                        f'{height:.3f}', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Fairness metrics visualization
        axes[1].axis('off')
        fairness_text = f"""Fairness Metrics:

AUC Difference: {fairness_metrics.get('auc_difference', 0):.4f}
AUC Ratio: {fairness_metrics.get('auc_ratio', 0):.4f}

Interpretation:
• AUC Difference < 0.1: Good fairness
• AUC Ratio > 0.8: Acceptable fairness

Group Performance:
"""
        
        for group, auc in fairness_metrics.get('group_aucs', {}).items():
            fairness_text += f"• {group}: {auc:.4f}\n"
        
        # Add fairness assessment
        auc_diff = fairness_metrics.get('auc_difference', 0)
        auc_ratio = fairness_metrics.get('auc_ratio', 0)
        
        if auc_diff < 0.1 and auc_ratio > 0.8:
            fairness_assessment = "✅ Model shows good fairness"
        elif auc_diff < 0.2 and auc_ratio > 0.7:
            fairness_assessment = "⚠️ Model shows moderate fairness"
        else:
            fairness_assessment = "❌ Model may have fairness issues"
        
        fairness_text += f"\nAssessment: {fairness_assessment}"
        
        axes[1].text(0.1, 0.9, fairness_text, transform=axes[1].transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        plt.show()