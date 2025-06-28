#!/usr/bin/env python3
"""
Complete Chest X-ray Disease Detection Training Pipeline
Advanced implementation with semi-supervised learning, uncertainty estimation, and comprehensive evaluation
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_generator import XrayDataGenerator
from model_architecture import ChestXrayModel
from training_pipeline import SemiSupervisedTrainer
from advanced_features import UncertaintyEstimator, GradCAMVisualizer, ModelEvaluator, FairnessAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CompleteTrainingPipeline:
    """Complete training pipeline with all advanced features"""
    
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        self.setup_gpu()
        
        # Initialize components
        self.data_generator = None
        self.model = None
        self.trainer = None
        self.evaluator = None
        self.uncertainty_estimator = None
        self.gradcam_visualizer = None
        self.fairness_analyzer = None
        
        # Results storage
        self.results = {}
        
    def setup_directories(self):
        """Setup required directories"""
        directories = [
            self.config['output_dir'],
            os.path.join(self.config['output_dir'], 'models'),
            os.path.join(self.config['output_dir'], 'plots'),
            os.path.join(self.config['output_dir'], 'logs'),
            os.path.join(self.config['output_dir'], 'results')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        logger.info(f"Created directories in: {self.config['output_dir']}")
    
    def setup_gpu(self):
        """Setup GPU configuration"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s)")
            except RuntimeError as e:
                logger.error(f"GPU setup error: {e}")
        else:
            logger.info("No GPU found, using CPU")
    
    def load_and_prepare_data(self):
        """Load and prepare datasets"""
        logger.info("Loading and preparing data...")
        
        # Initialize data generator
        self.data_generator = XrayDataGenerator(
            images_dir=self.config['images_dir'],
            batch_size=self.config['batch_size'],
            image_size=self.config['image_size'],
            augment=True
        )
        
        # Load datasets
        train_df = pd.read_csv(self.config['train_csv'])
        val_df = pd.read_csv(self.config['val_csv'])
        test_df = pd.read_csv(self.config['test_csv'])
        
        # Store datasets
        self.datasets = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        logger.info(f"Loaded datasets - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Create data generators
        self.train_generator = self.data_generator.create_generator(
            train_df, self.config['disease_labels'], shuffle=True
        )
        self.val_generator = self.data_generator.create_generator(
            val_df, self.config['disease_labels'], shuffle=False
        )
        self.test_generator = self.data_generator.create_generator(
            test_df, self.config['disease_labels'], shuffle=False
        )
        
        # Calculate steps per epoch
        self.steps_per_epoch = len(train_df) // self.config['batch_size']
        self.validation_steps = len(val_df) // self.config['batch_size']
        
        logger.info(f"Steps per epoch: {self.steps_per_epoch}, Validation steps: {self.validation_steps}")
        
        return self.datasets
    
    def build_model(self):
        """Build and compile the model"""
        logger.info("Building model...")
        
        model_builder = ChestXrayModel(
            num_classes=len(self.config['disease_labels']),
            input_shape=(*self.config['image_size'], 3),
            dropout_rate=self.config['dropout_rate']
        )
        
        self.model = model_builder.build_model()
        
        # Compile model
        optimizer = Adam(learning_rate=self.config['initial_lr'])
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        logger.info(f"Model built with {self.model.count_params():,} parameters")
        return self.model
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = os.path.join(self.config['output_dir'], 'models', 'best_model.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=self.config['patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        lr_reducer = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_reducer)
        
        # Custom callback for logging
        class TrainingLogger(keras.callbacks.Callback):
            def __init__(self, logger):
                self.logger = logger
                
            def on_epoch_end(self, epoch, logs=None):
                self.logger.info(
                    f"Epoch {epoch+1}: "
                    f"loss={logs['loss']:.4f}, "
                    f"val_loss={logs['val_loss']:.4f}, "
                    f"auc={logs['auc']:.4f}, "
                    f"val_auc={logs['val_auc']:.4f}"
                )
        
        callbacks.append(TrainingLogger(logger))
        
        return callbacks
    
    def train_model(self):
        """Train the model with multiple phases"""
        logger.info("Starting model training...")
        
        # Setup trainer
        self.trainer = SemiSupervisedTrainer(
            model=self.model,
            train_generator=self.train_generator,
            val_generator=self.val_generator,
            disease_labels=self.config['disease_labels']
        )
        
        # Phase 1: Initial supervised training
        logger.info("Phase 1: Initial supervised training")
        callbacks = self.setup_callbacks()
        
        history1 = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.config['initial_epochs'],
            validation_data=self.val_generator,
            validation_steps=self.validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tuning with lower learning rate
        logger.info("Phase 2: Fine-tuning with reduced learning rate")
        
        # Reduce learning rate
        self.model.optimizer.learning_rate = self.config['initial_lr'] * 0.1
        
        history2 = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.config['finetune_epochs'],
            validation_data=self.val_generator,
            validation_steps=self.validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 3: Semi-supervised learning (if unlabeled data available)
        if self.config.get('unlabeled_csv'):
            logger.info("Phase 3: Semi-supervised learning")
            
            unlabeled_df = pd.read_csv(self.config['unlabeled_csv'])
            history3 = self.trainer.semi_supervised_training(
                unlabeled_df=unlabeled_df,
                epochs=self.config['semisup_epochs'],
                confidence_threshold=self.config['confidence_threshold']
            )
            
            # Combine histories
            self.training_history = {
                'phase1': history1.history,
                'phase2': history2.history,
                'phase3': history3
            }
        else:
            self.training_history = {
                'phase1': history1.history,
                'phase2': history2.history
            }
        
        logger.info("Training completed successfully")
        return self.training_history
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        logger.info("Starting comprehensive model evaluation...")
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(self.model, self.data_generator)
        
        # Evaluate on test set
        test_results = self.evaluator.evaluate_model(
            self.datasets['test'], 
            self.config['disease_labels'],
            num_samples=self.config['eval_samples']
        )
        
        # Store results
        self.results['test_evaluation'] = test_results
        
        # Generate evaluation plots
        self.evaluator.plot_evaluation_results(test_results, self.config['disease_labels'])
        plt.savefig(os.path.join(self.config['output_dir'], 'plots', 'evaluation_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Test evaluation completed - Mean AUC: {test_results['mean_auc']:.4f}")
        return test_results
    
    def uncertainty_analysis(self):
        """Perform uncertainty analysis"""
        logger.info("Performing uncertainty analysis...")
        
        # Initialize uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(self.model)
        
        # Analyze uncertainty on test set
        uncertainty_results = self.uncertainty_estimator.analyze_uncertainty(
            self.datasets['test'], 
            self.data_generator,
            self.config['disease_labels'],
            num_samples=min(100, len(self.datasets['test']))
        )
        
        # Store results
        self.results['uncertainty_analysis'] = uncertainty_results
        
        # Generate uncertainty plots
        self.uncertainty_estimator.plot_uncertainty_analysis(uncertainty_results)
        plt.savefig(os.path.join(self.config['output_dir'], 'plots', 'uncertainty_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Uncertainty analysis completed")
        return uncertainty_results
    
    def gradcam_analysis(self):
        """Perform Grad-CAM analysis"""
        logger.info("Performing Grad-CAM analysis...")
        
        # Initialize Grad-CAM visualizer
        self.gradcam_visualizer = GradCAMVisualizer(self.model)
        
        # Generate Grad-CAM for sample images
        sample_df = self.datasets['test'].sample(n=min(5, len(self.datasets['test'])))
        
        gradcam_results = []
        for idx, row in sample_df.iterrows():
            img_path = os.path.join(self.config['images_dir'], row['Image Index'])
            
            # Generate Grad-CAM for top predicted disease
            img = self.data_generator.load_and_preprocess_image(img_path)
            if img is not None:
                pred = self.model.predict(np.expand_dims(img/255.0, axis=0), verbose=0)[0]
                top_class = np.argmax(pred)
                
                heatmap = self.gradcam_visualizer.generate_gradcam(
                    img_path, top_class, self.config['disease_labels'][top_class]
                )
                
                gradcam_results.append({
                    'image_path': img_path,
                    'predicted_disease': self.config['disease_labels'][top_class],
                    'confidence': pred[top_class],
                    'heatmap': heatmap
                })
        
        # Store results
        self.results['gradcam_analysis'] = gradcam_results
        
        # Generate Grad-CAM visualization
        self.gradcam_visualizer.plot_gradcam_results(gradcam_results)
        plt.savefig(os.path.join(self.config['output_dir'], 'plots', 'gradcam_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Grad-CAM analysis completed")
        return gradcam_results
    
    def fairness_analysis(self):
        """Perform fairness analysis"""
        logger.info("Performing fairness analysis...")
        
        # Initialize fairness analyzer
        self.fairness_analyzer = FairnessAnalyzer(self.model, self.data_generator)
        
        fairness_results = {}
        
        # Analyze by different demographic groups if available
        demographic_columns = ['Patient Gender', 'Patient Age']
        
        for col in demographic_columns:
            if col in self.datasets['test'].columns:
                logger.info(f"Analyzing fairness by {col}")
                
                group_results = self.fairness_analyzer.evaluate_by_group(
                    self.datasets['test'], col, self.config['disease_labels'],
                    num_samples=50
                )
                
                fairness_metrics = self.fairness_analyzer.calculate_fairness_metrics(group_results)
                
                fairness_results[col] = {
                    'group_results': group_results,
                    'fairness_metrics': fairness_metrics
                }
                
                # Generate fairness plots
                self.fairness_analyzer.plot_fairness_analysis(group_results, fairness_metrics)
                plt.savefig(os.path.join(self.config['output_dir'], 'plots', f'fairness_{col.lower().replace(" ", "_")}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        # Store results
        self.results['fairness_analysis'] = fairness_results
        
        logger.info("Fairness analysis completed")
        return fairness_results
    
    def generate_training_plots(self):
        """Generate training history plots"""
        logger.info("Generating training plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training History', fontsize=16)
        
        # Combine all training phases
        all_loss = []
        all_val_loss = []
        all_auc = []
        all_val_auc = []
        
        for phase, history in self.training_history.items():
            if isinstance(history, dict):
                all_loss.extend(history.get('loss', []))
                all_val_loss.extend(history.get('val_loss', []))
                all_auc.extend(history.get('auc', []))
                all_val_auc.extend(history.get('val_auc', []))
        
        epochs = range(1, len(all_loss) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, all_loss, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, all_val_loss, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUC plot
        axes[0, 1].plot(epochs, all_auc, 'b-', label='Training AUC', linewidth=2)
        axes[0, 1].plot(epochs, all_val_auc, 'r-', label='Validation AUC', linewidth=2)
        axes[0, 1].set_title('Model AUC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        if 'lr' in self.training_history.get('phase1', {}):
            all_lr = []
            for phase, history in self.training_history.items():
                if isinstance(history, dict):
                    all_lr.extend(history.get('lr', []))
            
            axes[1, 0].plot(epochs, all_lr, 'g-', linewidth=2)
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Learning Rate')
        
        # Training summary
        axes[1, 1].axis('off')
        summary_text = f"""Training Summary:

Total Epochs: {len(all_loss)}
Final Training Loss: {all_loss[-1]:.4f}
Final Validation Loss: {all_val_loss[-1]:.4f}
Final Training AUC: {all_auc[-1]:.4f}
Final Validation AUC: {all_val_auc[-1]:.4f}

Best Validation AUC: {max(all_val_auc):.4f}
Best Validation Loss: {min(all_val_loss):.4f}

Model Parameters: {self.model.count_params():,}
Training Time: {self.results.get('training_time', 'N/A')}
"""
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'plots', 'training_history.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Training plots generated")
    
    def save_results(self):
        """Save all results to files"""
        logger.info("Saving results...")
        
        results_dir = os.path.join(self.config['output_dir'], 'results')
        
        # Save training history
        with open(os.path.join(results_dir, 'training_history.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = {}
            for phase, history in self.training_history.items():
                if isinstance(history, dict):
                    serializable_history[phase] = {
                        k: [float(x) for x in v] if isinstance(v, (list, np.ndarray)) else v
                        for k, v in history.items()
                    }
                else:
                    serializable_history[phase] = history
            json.dump(serializable_history, f, indent=2)
        
        # Save evaluation results
        with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
            serializable_results = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in value.items() if not isinstance(v, np.ndarray)
                    }
                else:
                    serializable_results[key] = value
            json.dump(serializable_results, f, indent=2)
        
        # Save model summary
        with open(os.path.join(results_dir, 'model_summary.txt'), 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Save configuration
        with open(os.path.join(results_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Create final report
        self.create_final_report()
        
        logger.info(f"Results saved to {results_dir}")
    
    def create_final_report(self):
        """Create a comprehensive final report"""
        report_path = os.path.join(self.config['output_dir'], 'results', 'final_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Chest X-ray Disease Detection - Training Report\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration
            f.write("## Configuration\n\n")
            f.write("| Parameter | Value |\n")
            f.write("|-----------|-------|\n")
            for key, value in self.config.items():
                if not isinstance(value, (list, dict)):
                    f.write(f"| {key} | {value} |\n")
            f.write("\n")
            
            # Dataset Information
            f.write("## Dataset Information\n\n")
            f.write("| Split | Size |\n")
            f.write("|-------|------|\n")
            for split, df in self.datasets.items():
                f.write(f"| {split.capitalize()} | {len(df):,} |\n")
            f.write("\n")
            
            # Model Architecture
            f.write("## Model Architecture\n\n")
            f.write(f"- **Base Architecture:** ResNet50\n")
            f.write(f"- **Total Parameters:** {self.model.count_params():,}\n")
            f.write(f"- **Input Shape:** {self.config['image_size']}\n")
            f.write(f"- **Output Classes:** {len(self.config['disease_labels'])}\n")
            f.write(f"- **Dropout Rate:** {self.config['dropout_rate']}\n\n")
            
            # Training Results
            f.write("## Training Results\n\n")
            if hasattr(self, 'training_history'):
                # Get final metrics from last phase
                last_phase = list(self.training_history.keys())[-1]
                last_history = self.training_history[last_phase]
                
                if isinstance(last_history, dict):
                    f.write("| Metric | Final Value |\n")
                    f.write("|--------|-------------|\n")
                    for metric in ['loss', 'val_loss', 'auc', 'val_auc']:
                        if metric in last_history:
                            f.write(f"| {metric} | {last_history[metric][-1]:.4f} |\n")
                f.write("\n")
            
            # Evaluation Results
            f.write("## Evaluation Results\n\n")
            if 'test_evaluation' in self.results:
                eval_results = self.results['test_evaluation']
                f.write(f"**Mean AUC:** {eval_results.get('mean_auc', 'N/A'):.4f}\n\n")
                
                if 'disease_aucs' in eval_results:
                    f.write("### Disease-specific AUC Scores\n\n")
                    f.write("| Disease | AUC |\n")
                    f.write("|---------|-----|\n")
                    for disease, auc in eval_results['disease_aucs'].items():
                        f.write(f"| {disease} | {auc:.4f} |\n")
                    f.write("\n")
            
            # Uncertainty Analysis
            if 'uncertainty_analysis' in self.results:
                f.write("## Uncertainty Analysis\n\n")
                uncertainty_results = self.results['uncertainty_analysis']
                f.write(f"**Mean Uncertainty:** {uncertainty_results.get('mean_uncertainty', 'N/A'):.4f}\n")
                f.write(f"**Uncertainty Std:** {uncertainty_results.get('uncertainty_std', 'N/A'):.4f}\n\n")
            
            # Fairness Analysis
            if 'fairness_analysis' in self.results:
                f.write("## Fairness Analysis\n\n")
                fairness_results = self.results['fairness_analysis']
                for demographic, results in fairness_results.items():
                    f.write(f"### {demographic}\n\n")
                    if 'fairness_metrics' in results:
                        metrics = results['fairness_metrics']
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                f.write(f"| {metric} | {value:.4f} |\n")
                    f.write("\n")
            
            # Files Generated
            f.write("## Generated Files\n\n")
            f.write("- `models/best_model.h5` - Best trained model\n")
            f.write("- `plots/training_history.png` - Training curves\n")
            f.write("- `plots/evaluation_results.png` - Evaluation metrics\n")
            f.write("- `plots/uncertainty_analysis.png` - Uncertainty analysis\n")
            f.write("- `plots/gradcam_analysis.png` - Grad-CAM visualizations\n")
            f.write("- `results/training_history.json` - Training history data\n")
            f.write("- `results/evaluation_results.json` - Evaluation results\n")
            f.write("- `results/model_summary.txt` - Model architecture summary\n")
            f.write("- `results/config.json` - Training configuration\n")
            f.write("- `logs/training.log` - Training logs\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            if 'test_evaluation' in self.results:
                mean_auc = self.results['test_evaluation'].get('mean_auc', 0)
                if mean_auc > 0.85:
                    f.write("✅ **Excellent Performance:** Model shows strong diagnostic capability.\n")
                elif mean_auc > 0.80:
                    f.write("✅ **Good Performance:** Model performs well but could benefit from further optimization.\n")
                else:
                    f.write("⚠️ **Needs Improvement:** Consider data augmentation, architecture changes, or more training data.\n")
            
            if 'uncertainty_analysis' in self.results:
                mean_uncertainty = self.results['uncertainty_analysis'].get('mean_uncertainty', 0)
                if mean_uncertainty < 0.1:
                    f.write("✅ **Well-calibrated:** Model provides reliable confidence estimates.\n")
                else:
                    f.write("⚠️ **High Uncertainty:** Consider uncertainty calibration techniques.\n")
            
            f.write("\n---\n")
            f.write("*Report generated by Chest X-ray AI Training Pipeline*\n")
        
        logger.info(f"Final report saved to {report_path}")
    
    def run_complete_pipeline(self):
        """Run the complete training and evaluation pipeline"""
        start_time = datetime.now()
        logger.info("Starting complete training pipeline...")
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()
            
            # Step 2: Build model
            self.build_model()
            
            # Step 3: Train model
            self.train_model()
            
            # Step 4: Evaluate model
            self.evaluate_model()
            
            # Step 5: Uncertainty analysis
            self.uncertainty_analysis()
            
            # Step 6: Grad-CAM analysis
            self.gradcam_analysis()
            
            # Step 7: Fairness analysis
            self.fairness_analysis()
            
            # Step 8: Generate plots
            self.generate_training_plots()
            
            # Step 9: Save results
            end_time = datetime.now()
            self.results['training_time'] = str(end_time - start_time)
            self.save_results()
            
            logger.info(f"Pipeline completed successfully in {end_time - start_time}")
            logger.info(f"Results saved to: {self.config['output_dir']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise e

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_default_config():
    """Create default configuration"""
    return {
        # Data paths
        "images_dir": "data/images",
        "train_csv": "data/train_list.csv",
        "val_csv": "data/val_list.csv",
        "test_csv": "data/test_list.csv",
        "unlabeled_csv": None,  # Optional for semi-supervised learning
        
        # Output directory
        "output_dir": f"experiments/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        
        # Model parameters
        "image_size": [224, 224],
        "batch_size": 32,
        "dropout_rate": 0.5,
        
        # Training parameters
        "initial_lr": 0.001,
        "initial_epochs": 10,
        "finetune_epochs": 5,
        "semisup_epochs": 3,
        "patience": 7,
        "confidence_threshold": 0.8,
        
        # Evaluation parameters
        "eval_samples": 1000,
        
        # Disease labels
        "disease_labels": [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
            "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
        ]
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Complete Chest X-ray Training Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--images-dir', type=str, help='Images directory')
    parser.add_argument('--train-csv', type=str, help='Training CSV file')
    parser.add_argument('--val-csv', type=str, help='Validation CSV file')
    parser.add_argument('--test-csv', type=str, help='Test CSV file')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Total epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = create_default_config()
        logger.info("Using default configuration")
    
    # Override config with command line arguments
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.images_dir:
        config['images_dir'] = args.images_dir
    if args.train_csv:
        config['train_csv'] = args.train_csv
    if args.val_csv:
        config['val_csv'] = args.val_csv
    if args.test_csv:
        config['test_csv'] = args.test_csv
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['initial_epochs'] = args.epochs
    if args.lr:
        config['initial_lr'] = args.lr
    
    # Validate required paths
    required_paths = ['images_dir', 'train_csv', 'val_csv', 'test_csv']
    for path_key in required_paths:
        if not os.path.exists(config[path_key]):
            logger.error(f"Required path does not exist: {config[path_key]}")
            sys.exit(1)
    
    # Print configuration
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize and run pipeline
    pipeline = CompleteTrainingPipeline(config)
    success = pipeline.run_complete_pipeline()
    
    if success:
        logger.info("Training pipeline completed successfully!")
        print(f"\n{'='*60}")
        print("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print(f"{'='*60}")
        print(f"📁 Results saved to: {config['output_dir']}")
        print(f"📊 Check the final report: {config['output_dir']}/results/final_report.md")
        print(f"🤖 Best model saved: {config['output_dir']}/models/best_model.h5")
        print(f"📈 Training plots: {config['output_dir']}/plots/")
        print(f"📋 Logs available: {config['output_dir']}/logs/training.log")
        print(f"{'='*60}")
    else:
        logger.error("Training pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()