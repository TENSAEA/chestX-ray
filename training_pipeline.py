import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import tensorflow as tf

class TrainingPipeline:
    def __init__(self, model, data_generator, disease_labels):
        self.model = model
        self.data_generator = data_generator
        self.disease_labels = disease_labels
        self.history = {}
        
    def train_initial(self, train_generator, val_generator, train_steps, val_steps, epochs=3):
        """Initial training with frozen backbone"""
        print("=== Stage 1: Initial Training (Frozen Backbone) ===")
        
        callbacks = ChestXrayModel().get_callbacks('models/initial_model.h5')
        
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history['initial'] = history.history
        return history
    
    def fine_tune(self, train_generator, val_generator, train_steps, val_steps, epochs=2):
        """Fine-tuning with unfrozen top layers"""
        print("=== Stage 2: Fine-tuning (Unfrozen Top Layers) ===")
        
        # Unfreeze top layers
        model_builder = ChestXrayModel()
        self.model = model_builder.unfreeze_top_layers(self.model, num_layers=30)
        
        callbacks = model_builder.get_callbacks('models/finetuned_model.h5')
        
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history['fine_tune'] = history.history
        return history
    
    def generate_pseudo_labels(self, unlabeled_df, confidence_threshold=0.8):
        """Generate pseudo-labels for semi-supervised learning"""
        print("Generating pseudo-labels...")
        
        pseudo_labeled_data = []
        
        for _, row in unlabeled_df.iterrows():
            img_path = os.path.join(self.data_generator.images_dir, row['Image Index'])
            img = self.data_generator.load_and_preprocess_image(img_path)
            
            if img is not None:
                img_normalized = img / 255.0
                img_batch = np.expand_dims(img_normalized, axis=0)
                
                # Get prediction
                pred = self.model.predict(img_batch, verbose=0)[0]
                
                # Check if any prediction is confident enough
                max_confidence = np.max(pred)
                if max_confidence > confidence_threshold or np.min(pred) < (1 - confidence_threshold):
                    # Create pseudo-labeled row
                    new_row = row.copy()
                    for i, disease in enumerate(self.disease_labels):
                        new_row[disease] = 1 if pred[i] > 0.5 else 0
                    pseudo_labeled_data.append(new_row)
        
        if pseudo_labeled_data:
            pseudo_df = pd.DataFrame(pseudo_labeled_data)
            print(f"Generated {len(pseudo_df)} pseudo-labeled samples")
            return pseudo_df
        else:
            print("No confident pseudo-labels generated")
            return pd.DataFrame()
    
    def semi_supervised_training(self, labeled_df, unlabeled_df, val_df, epochs=2):
        """Semi-supervised learning with pseudo-labels"""
        print("=== Stage 3: Semi-supervised Learning ===")
        
        # Generate pseudo-labels
        pseudo_df = self.generate_pseudo_labels(unlabeled_df, confidence_threshold=0.7)
        
        if len(pseudo_df) > 0:
            # Combine labeled and pseudo-labeled data
            combined_df = pd.concat([labeled_df, pseudo_df], ignore_index=True)
            print(f"Combined dataset size: {len(combined_df)}")
            
            # Create new generators
            combined_gen = self.data_generator.create_generator(
                combined_df, self.disease_labels, is_training=True
            )
            val_gen = self.data_generator.create_generator(
                val_df, self.disease_labels, is_training=False
            )
            
            combined_steps = self.data_generator.get_steps_per_epoch(len(combined_df))
            val_steps = self.data_generator.get_steps_per_epoch(len(val_df))
            
            callbacks = ChestXrayModel().get_callbacks('models/semi_supervised_model.h5')
            
            # Train with combined data
            history = self.model.fit(
                combined_gen,
                steps_per_epoch=combined_steps,
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=val_steps,
                callbacks=callbacks,
                verbose=1
            )
            
            self.history['semi_supervised'] = history.history
            return history
        else:
            print("Skipping semi-supervised training - no pseudo-labels generated")
            return None
    
    def plot_training_history(self):
        """Plot comprehensive training history"""
        stages = ['initial', 'fine_tune', 'semi_supervised']
        colors = ['blue', 'red', 'green']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        for stage, color in zip(stages, colors):
            if stage in self.history:
                hist = self.history[stage]
                epochs_offset = sum(len(self.history[s]['loss']) for s in stages[:stages.index(stage)] if s in self.history)
                epochs = range(epochs_offset, epochs_offset + len(hist['loss']))
                
                # Loss
                axes[0, 0].plot(epochs, hist['loss'], color=color, label=f'{stage}_loss', linewidth=2)
                axes[0, 0].plot(epochs, hist['val_loss'], color=color, linestyle='--', label=f'{stage}_val_loss', linewidth=2)
                
                # AUC
                axes[0, 1].plot(epochs, hist['auc'], color=color, label=f'{stage}_auc', linewidth=2)
                axes[0, 1].plot(epochs, hist['val_auc'], color=color, linestyle='--', label=f'{stage}_val_auc', linewidth=2)
                
                # Binary Accuracy
                axes[1, 0].plot(epochs, hist['binary_accuracy'], color=color, label=f'{stage}_acc', linewidth=2)
                axes[1, 0].plot(epochs, hist['val_binary_accuracy'], color=color, linestyle='--', label=f'{stage}_val_acc', linewidth=2)
        
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Training AUC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training summary
        axes[1, 1].axis('off')
        summary_text = "Training Summary:\n\n"
        for stage in stages:
            if stage in self.history:
                hist = self.history[stage]
                final_loss = hist['val_loss'][-1]
                final_auc = hist['val_auc'][-1]
                summary_text += f"{stage.replace('_', ' ').title()}:\n"
                summary_text += f"  Final Val Loss: {final_loss:.4f}\n"
                summary_text += f"  Final Val AUC: {final_auc:.4f}\n\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()