#!/usr/bin/env python3
"""
Standalone model evaluation script for trained chest X-ray models
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, average_precision_score
)
import tensorflow as tf
from tensorflow import keras
import json
from datetime import datetime

# Import custom modules
from data_generator import XrayDataGenerator
from advanced_features import ModelEvaluator, UncertaintyEstimator, GradCAMVisualizer

class StandaloneEvaluator:
    """Standalone model evaluator for trained models"""
    
    def __init__(self, model_path, data_dir, output_dir, batch_size=32):
        self.model_path = model_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Disease labels (standard NIH dataset)
        self.disease_labels = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
            "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
        ]
        
        # Load model
        self.model = self.load_model()
        
        # Initialize data generator
        self.data_generator = XrayDataGenerator(
            images_dir=os.path.join(data_dir, 'images'),
            batch_size=batch_size,
            image_size=(224, 224),
            augment=False  # No augmentation for evaluation
        )
        
        # Results storage
        self.results = {}
    
    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from: {self.model_path}")
        
        try:
            model = keras.models.load_model(self.model_path)
            print(f"✅ Model loaded successfully")
            print(f"Model input shape: {model.input_shape}")
            print(f"Model output shape: {model.output_shape}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def load_test_data(self):
        """Load test dataset"""
        test_csv_path = os.path.join(self.data_dir, 'test_list.csv')
        
        if not os.path.exists(test_csv_path):
            print(f"❌ Test CSV not found: {test_csv_path}")
            sys.exit(1)
        
        test_df = pd.read_csv(test_csv_path)
        print(f"✅ Loaded test dataset: {len(test_df)} samples")
        
        return test_df
    
    def evaluate_model_performance(self, test_df):
        """Evaluate model performance on test set"""
        print("📊 Evaluating model performance...")
        
        # Create test generator
        test_generator = self.data_generator.create_generator(
            test_df, self.disease_labels, shuffle=False
        )
        
        # Get predictions
        print("🔮 Generating predictions...")
        y_true = []
        y_pred = []
        
        steps = len(test_df) // self.batch_size
        for i, (batch_x, batch_y) in enumerate(test_generator):
            if i >= steps:
                break
                
            pred = self.model.predict(batch_x, verbose=0)
            y_true.append(batch_y)
            y_pred.append(pred)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{steps} batches")
        
        # Concatenate results
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        
        print(f"✅ Generated predictions for {len(y_true)} samples")
        
        # Calculate metrics
        results = self.calculate_metrics(y_true, y_pred)
        
        # Store results
        self.results['performance'] = results
        
        return results
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        print("📈 Calculating metrics...")
        
        results = {
            'num_samples': len(y_true),
            'disease_metrics': {},
            'overall_metrics': {}
        }
        
        # Per-disease metrics
        disease_aucs = []
        disease_aps = []
        
        for i, disease in enumerate(self.disease_labels):
            # AUC-ROC
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                disease_aucs.append(auc)
            except ValueError:
                auc = 0.5  # No positive samples
                disease_aucs.append(auc)
            
            # Average Precision
            try:
                ap = average_precision_score(y_true[:, i], y_pred[:, i])
                disease_aps.append(ap)
            except ValueError:
                ap = 0.0
                disease_aps.append(ap)
            
            results['disease_metrics'][disease] = {
                'auc': float(auc),
                'average_precision': float(ap),
                'prevalence': float(np.mean(y_true[:, i]))
            }
        
        # Overall metrics
        results['overall_metrics'] = {
            'mean_auc': float(np.mean(disease_aucs)),
            'std_auc': float(np.std(disease_aucs)),
            'mean_ap': float(np.mean(disease_aps)),
            'std_ap': float(np.std(disease_aps)),
            'min_auc': float(np.min(disease_aucs)),
            'max_auc': float(np.max(disease_aucs))
        }
        
        print(f"✅ Mean AUC: {results['overall_metrics']['mean_auc']:.4f}")
        print(f"✅ Mean AP: {results['overall_metrics']['mean_ap']:.4f}")
        
        return results
    
    def generate_evaluation_plots(self, y_true, y_pred):
        """Generate comprehensive evaluation plots"""
        print("📊 Generating evaluation plots...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. ROC Curves
        self.plot_roc_curves(y_true, y_pred)
        
        # 2. Precision-Recall Curves
        self.plot_pr_curves(y_true, y_pred)
        
        # 3. Performance Summary
        self.plot_performance_summary()
        
        # 4. Disease Distribution
        self.plot_disease_distribution(y_true)
        
        print("✅ Evaluation plots generated")
    
    def plot_roc_curves(self, y_true, y_pred):
        """Plot ROC curves for all diseases"""
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.ravel()
        
        for i, disease in enumerate(self.disease_labels):
            if i < len(axes):
                fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                
                axes[i].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
                axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5)
                axes[i].set_xlim([0.0, 1.0])
                axes[i].set_ylim([0.0, 1.05])
                axes[i].set_xlabel('False Positive Rate')
                axes[i].set_ylabel('True Positive Rate')
                axes[i].set_title(f'{disease}')
                axes[i].legend(loc="lower right")
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(self.disease_labels), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('ROC Curves for All Diseases', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pr_curves(self, y_true, y_pred):
        """Plot Precision-Recall curves for all diseases"""
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.ravel()
        
        for i, disease in enumerate(self.disease_labels):
            if i < len(axes):
                precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
                ap = average_precision_score(y_true[:, i], y_pred[:, i])
                
                axes[i].plot(recall, precision, linewidth=2, label=f'AP = {ap:.3f}')
                axes[i].set_xlim([0.0, 1.0])
                axes[i].set_ylim([0.0, 1.05])
                axes[i].set_xlabel('Recall')
                axes[i].set_ylabel('Precision')
                axes[i].set_title(f'{disease}')
                axes[i].legend(loc="lower left")
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(self.disease_labels), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Precision-Recall Curves for All Diseases', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pr_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_summary(self):
        """Plot performance summary"""
        if 'performance' not in self.results:
            return
        
        disease_metrics = self.results['performance']['disease_metrics']
        
        # Extract metrics
        diseases = list(disease_metrics.keys())
        aucs = [disease_metrics[d]['auc'] for d in diseases]
        aps = [disease_metrics[d]['average_precision'] for d in diseases]
        
        # Create summary plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # AUC scores
        y_pos = np.arange(len(diseases))
        bars1 = ax1.barh(y_pos, aucs, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(diseases)
        ax1.set_xlabel('AUC Score')
        ax1.set_title('AUC Scores by Disease')
        ax1.set_xlim([0, 1])
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, auc) in enumerate(zip(bars1, aucs)):
            ax1.text(auc + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{auc:.3f}', va='center', fontsize=10)
        
        # Average Precision scores
        bars2 = ax2.barh(y_pos, aps, alpha=0.8, color='orange')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(diseases)
        ax2.set_xlabel('Average Precision')
        ax2.set_title('Average Precision by Disease')
        ax2.set_xlim([0, 1])
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, ap) in enumerate(zip(bars2, aps)):
            ax2.text(ap + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{ap:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_disease_distribution(self, y_true):
        """Plot disease distribution in test set"""
        disease_counts = np.sum(y_true, axis=0)
        total_samples = len(y_true)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Absolute counts
        y_pos = np.arange(len(self.disease_labels))
        bars1 = ax1.barh(y_pos, disease_counts, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(self.disease_labels)
        ax1.set_xlabel('Number of Cases')
        ax1.set_title('Disease Distribution (Absolute Counts)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars1, disease_counts)):
            ax1.text(count + max(disease_counts)*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{int(count)}', va='center', fontsize=10)
        
        # Percentages
        disease_percentages = (disease_counts / total_samples) * 100
        bars2 = ax2.barh(y_pos, disease_percentages, alpha=0.8, color='green')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(self.disease_labels)
        ax2.set_xlabel('Percentage of Cases (%)')
        ax2.set_title('Disease Distribution (Percentages)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, pct) in enumerate(zip(bars2, disease_percentages)):
            ax2.text(pct + max(disease_percentages)*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{pct:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'disease_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def uncertainty_analysis(self, test_df):
        """Perform uncertainty analysis"""
        print("🎲 Performing uncertainty analysis...")
        
        try:
            uncertainty_estimator = UncertaintyEstimator(self.model)
            
            uncertainty_results = uncertainty_estimator.analyze_uncertainty(
                test_df.sample(n=min(100, len(test_df))),  # Sample for efficiency
                self.data_generator,
                self.disease_labels,
                num_samples=50
            )
            
            self.results['uncertainty'] = uncertainty_results
            
            # Generate uncertainty plots
            uncertainty_estimator.plot_uncertainty_analysis(uncertainty_results)
            plt.savefig(os.path.join(self.output_dir, 'uncertainty_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Uncertainty analysis completed")
            
        except Exception as e:
            print(f"⚠️ Uncertainty analysis failed: {e}")
    
    def gradcam_analysis(self, test_df):
        """Perform Grad-CAM analysis"""
        print("🔍 Performing Grad-CAM analysis...")
        
        try:
            gradcam_visualizer = GradCAMVisualizer(self.model)
            
            # Sample a few images for Grad-CAM
            sample_df = test_df.sample(n=min(5, len(test_df)))
            
            gradcam_results = []
            for idx, row in sample_df.iterrows():
                img_path = os.path.join(self.data_generator.images_dir, row['Image Index'])
                
                if os.path.exists(img_path):
                    # Load and predict
                    img = self.data_generator.load_and_preprocess_image(img_path)
                    if img is not None:
                        pred = self.model.predict(np.expand_dims(img/255.0, axis=0), verbose=0)[0]
                        top_class = np.argmax(pred)
                        
                        # Generate Grad-CAM
                        heatmap = gradcam_visualizer.generate_gradcam(
                            img_path, top_class, self.disease_labels[top_class]
                        )
                        
                        gradcam_results.append({
                            'image_path': img_path,
                            'predicted_disease': self.disease_labels[top_class],
                            'confidence': float(pred[top_class]),
                            'heatmap': heatmap
                        })
            
            if gradcam_results:
                self.results['gradcam'] = gradcam_results
                
                # Generate Grad-CAM visualization
                gradcam_visualizer.plot_gradcam_results(gradcam_results)
                plt.savefig(os.path.join(self.output_dir, 'gradcam_analysis.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print("✅ Grad-CAM analysis completed")
            else:
                print("⚠️ No valid images found for Grad-CAM analysis")
                
        except Exception as e:
            print(f"⚠️ Grad-CAM analysis failed: {e}")
    
    def save_results(self):
        """Save evaluation results"""
        print("💾 Saving results...")
        
        # Save detailed results as JSON
        results_file = os.path.join(self.output_dir, 'evaluation_results.json')
        
        # Make results JSON serializable
        serializable_results = {}
        for key, value in self.results.items():
            if key == 'gradcam':
                # Skip gradcam results as they contain numpy arrays
                continue
            elif isinstance(value, dict):
                serializable_results[key] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in value.items() if not isinstance(v, np.ndarray)
                }
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save summary report
        self.generate_summary_report()
        
        print(f"✅ Results saved to {self.output_dir}")
    
    def generate_summary_report(self):
        """Generate a summary report"""
        report_file = os.path.join(self.output_dir, 'evaluation_report.md')
        
        with open(report_file, 'w') as f:
            f.write("# Model Evaluation Report\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model Information
            f.write("## Model Information\n\n")
            f.write(f"- **Model Path:** {self.model_path}\n")
            f.write(f"- **Model Input Shape:** {self.model.input_shape}\n")
            f.write(f"- **Model Output Shape:** {self.model.output_shape}\n")
            f.write(f"- **Total Parameters:** {self.model.count_params():,}\n\n")
            
            # Dataset Information
            if 'performance' in self.results:
                perf = self.results['performance']
                f.write("## Dataset Information\n\n")
                f.write(f"- **Test Samples:** {perf['num_samples']:,}\n")
                f.write(f"- **Number of Diseases:** {len(self.disease_labels)}\n\n")
                
                # Overall Performance
                f.write("## Overall Performance\n\n")
                overall = perf['overall_metrics']
                f.write(f"- **Mean AUC:** {overall['mean_auc']:.4f} ± {overall['std_auc']:.4f}\n")
                f.write(f"- **Mean Average Precision:** {overall['mean_ap']:.4f} ± {overall['std_ap']:.4f}\n")
                f.write(f"- **AUC Range:** {overall['min_auc']:.4f} - {overall['max_auc']:.4f}\n\n")
                
                # Per-Disease Performance
                f.write("## Per-Disease Performance\n\n")
                f.write("| Disease | AUC | Average Precision | Prevalence |\n")
                f.write("|---------|-----|-------------------|------------|\n")
                
                disease_metrics = perf['disease_metrics']
                for disease in self.disease_labels:
                    if disease in disease_metrics:
                        metrics = disease_metrics[disease]
                        f.write(f"| {disease} | {metrics['auc']:.4f} | {metrics['average_precision']:.4f} | {metrics['prevalence']:.4f} |\n")
                f.write("\n")
            
            # Uncertainty Analysis
            if 'uncertainty' in self.results:
                f.write("## Uncertainty Analysis\n\n")
                uncertainty = self.results['uncertainty']
                f.write(f"- **Mean Uncertainty:** {uncertainty.get('mean_uncertainty', 'N/A'):.4f}\n")
                f.write(f"- **Uncertainty Std:** {uncertainty.get('uncertainty_std', 'N/A'):.4f}\n\n")
            
            # Generated Files
            f.write("## Generated Files\n\n")
            f.write("- `evaluation_results.json` - Detailed evaluation metrics\n")
            f.write("- `roc_curves.png` - ROC curves for all diseases\n")
            f.write("- `pr_curves.png` - Precision-Recall curves\n")
            f.write("- `performance_summary.png` - Performance summary charts\n")
            f.write("- `disease_distribution.png` - Disease distribution in test set\n")
            f.write("- `uncertainty_analysis.png` - Uncertainty analysis (if available)\n")
            f.write("- `gradcam_analysis.png` - Grad-CAM visualizations (if available)\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            if 'performance' in self.results:
                mean_auc = self.results['performance']['overall_metrics']['mean_auc']
                if mean_auc > 0.85:
                    f.write("✅ **Excellent Performance:** The model demonstrates strong diagnostic capability across diseases.\n\n")
                elif mean_auc > 0.80:
                    f.write("✅ **Good Performance:** The model performs well but may benefit from further optimization.\n\n")
                elif mean_auc > 0.70:
                    f.write("⚠️ **Moderate Performance:** Consider model improvements or additional training data.\n\n")
                else:
                    f.write("❌ **Poor Performance:** Significant improvements needed. Consider architecture changes or data quality issues.\n\n")
                
                # Disease-specific recommendations
                disease_metrics = self.results['performance']['disease_metrics']
                low_performing_diseases = [
                    disease for disease, metrics in disease_metrics.items()
                    if metrics['auc'] < 0.70
                ]
                
                if low_performing_diseases:
                    f.write("### Diseases Requiring Attention\n\n")
                    f.write("The following diseases show lower performance and may need additional focus:\n\n")
                    for disease in low_performing_diseases:
                        auc = disease_metrics[disease]['auc']
                        prevalence = disease_metrics[disease]['prevalence']
                        f.write(f"- **{disease}:** AUC = {auc:.3f}, Prevalence = {prevalence:.3f}\n")
                    f.write("\n")
            
            f.write("---\n")
            f.write("*Report generated by Chest X-ray AI Evaluation Pipeline*\n")
    
    def run_complete_evaluation(self):
        """Run complete evaluation pipeline"""
        print("🚀 Starting complete model evaluation...")
        start_time = datetime.now()
        
        try:
            # Load test data
            test_df = self.load_test_data()
            
            # Evaluate model performance
            y_true, y_pred = self.get_predictions(test_df)
            self.evaluate_model_performance_with_data(y_true, y_pred)
            
            # Generate plots
            self.generate_evaluation_plots(y_true, y_pred)
            
            # Uncertainty analysis
            self.uncertainty_analysis(test_df)
            
            # Grad-CAM analysis
            self.gradcam_analysis(test_df)
            
            # Save results
            self.save_results()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print(f"\n{'='*60}")
            print("🎉 EVALUATION COMPLETED SUCCESSFULLY! 🎉")
            print(f"{'='*60}")
            print(f"⏱️  Duration: {duration}")
            print(f"📁 Results saved to: {self.output_dir}")
            print(f"📊 Summary report: {self.output_dir}/evaluation_report.md")
            print(f"📈 Plots available in: {self.output_dir}/")
            print(f"{'='*60}")
            
            return True
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return False
    
    def get_predictions(self, test_df):
        """Get model predictions for test data"""
        print("🔮 Generating predictions...")
        
        # Create test generator
        test_generator = self.data_generator.create_generator(
            test_df, self.disease_labels, shuffle=False
        )
        
        y_true = []
        y_pred = []
        
        steps = len(test_df) // self.batch_size
        for i, (batch_x, batch_y) in enumerate(test_generator):
            if i >= steps:
                break
                
            pred = self.model.predict(batch_x, verbose=0)
            y_true.append(batch_y)
            y_pred.append(pred)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{steps} batches")
        
        # Concatenate results
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        
        print(f"✅ Generated predictions for {len(y_true)} samples")
        return y_true, y_pred
    
    def evaluate_model_performance_with_data(self, y_true, y_pred):
        """Evaluate model performance with prediction data"""
        results = self.calculate_metrics(y_true, y_pred)
        self.results['performance'] = results
        return results

def evaluate_trained_model(model_path, data_dir, output_dir, batch_size=32):
    """Convenience function for evaluating a trained model"""
    evaluator = StandaloneEvaluator(model_path, data_dir, output_dir, batch_size)
    return evaluator.run_complete_evaluation()

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Evaluate trained chest X-ray model')
    parser.add_argument('--model', required=True, help='Path to trained model file')
    parser.add_argument('--data-dir', default='data', help='Data directory containing test_list.csv and images/')
    parser.add_argument('--output-dir', default='evaluation_results', help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    test_csv = os.path.join(args.data_dir, 'test_list.csv')
    if not os.path.exists(test_csv):
        print(f"❌ Test CSV not found: {test_csv}")
        sys.exit(1)
    
    images_dir = os.path.join(args.data_dir, 'images')
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        sys.exit(1)
    
    # Run evaluation
    success = evaluate_trained_model(
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()