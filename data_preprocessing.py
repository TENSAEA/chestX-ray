import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class ChestXrayDataProcessor:
    def __init__(self, csv_path, images_dir):
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.disease_labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
    
    def load_and_filter_data(self):
        """Load and filter the dataset"""
        print("Loading dataset...")
        df = pd.read_csv(self.csv_path)
        
        # Filter out images that don't exist
        existing_images = []
        for idx, row in df.iterrows():
            img_path = os.path.join(self.images_dir, row['Image Index'])
            if os.path.exists(img_path):
                existing_images.append(idx)
        
        df_filtered = df.loc[existing_images].reset_index(drop=True)
        print(f"Loaded {len(df_filtered)} samples with existing images")
        
        return df_filtered
    
    def preprocess_labels(self, df):
        """Preprocess labels for multi-label classification"""
        print("Preprocessing labels...")
        
        # Ensure all disease columns exist
        for disease in self.disease_labels:
            if disease not in df.columns:
                df[disease] = 0
        
        return df
    
    def analyze_dataset(self, df):
        """Analyze dataset distribution"""
        print("\n=== Dataset Analysis ===")
        print(f"Total samples: {len(df)}")
        print(f"Unique patients: {df['Patient ID'].nunique()}")
        
        # Disease distribution
        disease_counts = {}
        for disease in self.disease_labels:
            count = df[disease].sum()
            disease_counts[disease] = count
            print(f"{disease}: {count} ({count/len(df)*100:.1f}%)")
        
        # Gender distribution
        if 'Patient Gender' in df.columns:
            print(f"\nGender distribution:")
            print(df['Patient Gender'].value_counts())
        
        return disease_counts
    
    def split_data(self, df, test_size=0.2, val_size=0.1):
        """Split data into train/val/test sets"""
        print("\nSplitting data...")
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df['Patient Gender']
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size_adjusted, random_state=42, 
            stratify=train_val_df['Patient Gender']
        )
        
        # Create labeled and unlabeled splits for semi-supervised learning
        # Use 70% of training data as labeled, 30% as unlabeled
        labeled_train, unlabeled_train = train_test_split(
            train_df, test_size=0.3, random_state=42
        )
        
        print(f"Labeled training: {len(labeled_train)}")
        print(f"Unlabeled training: {len(unlabeled_train)}")
        print(f"Validation: {len(val_df)}")
        print(f"Test: {len(test_df)}")
        
        return labeled_train, unlabeled_train, val_df, test_df