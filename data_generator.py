import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import os

class XrayDataGenerator(Sequence):
    def __init__(self, images_dir, batch_size=16, image_size=(224, 224), 
                 is_training=False, augment=True):
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.is_training = is_training
        self.augment = augment and is_training
        
    def create_generator(self, df, disease_labels, is_training=False):
        """Create a data generator"""
        return XrayBatchGenerator(
            df, disease_labels, self.images_dir, self.batch_size, 
            self.image_size, is_training, self.augment
        )
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            # Resize
            img = cv2.resize(img, self.image_size)
            
            # Convert to RGB (3 channels) for ResNet50
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            return img_rgb
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def augment_image(self, image):
        """Apply data augmentation"""
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            center = (image.shape[1]//2, image.shape[0]//2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        # Random zoom
        if np.random.random() > 0.7:
            zoom_factor = np.random.uniform(0.9, 1.1)
            h, w = image.shape[:2]
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            
            if zoom_factor > 1:
                # Zoom in - crop center
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                image = cv2.resize(image, (new_w, new_h))
                image = image[start_h:start_h+h, start_w:start_w+w]
            else:
                # Zoom out - pad
                image = cv2.resize(image, (new_w, new_h))
                image = cv2.resize(image, (w, h))
        
        return image
    
    def get_steps_per_epoch(self, num_samples):
        """Calculate steps per epoch"""
        return max(1, num_samples // self.batch_size)

class XrayBatchGenerator(Sequence):
    def __init__(self, df, disease_labels, images_dir, batch_size, 
                 image_size, is_training, augment):
        self.df = df.reset_index(drop=True)
        self.disease_labels = disease_labels
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.is_training = is_training
        self.augment = augment
        self.data_generator = XrayDataGenerator(images_dir, batch_size, image_size)
        
        if is_training:
            self.on_epoch_end()
    
    def __len__(self):
        return max(1, len(self.df) // self.batch_size)
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return self._generate_batch(batch_indices)
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.is_training:
            np.random.shuffle(self.indices)
    
    def _generate_batch(self, batch_indices):
        batch_images = []
        batch_labels = []
        
        for idx in batch_indices:
            if idx >= len(self.df):
                continue
                
            row = self.df.iloc[idx]
            img_path = os.path.join(self.images_dir, row['Image Index'])
            
            # Load image
            img = self.data_generator.load_and_preprocess_image(img_path)
            if img is None:
                # Create a dummy image if loading fails
                img = np.zeros((*self.image_size, 3), dtype=np.uint8)
            
            # Apply augmentation
            if self.augment:
                img = self.data_generator.augment_image(img)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            # Get labels
            labels = row[self.disease_labels].values.astype(np.float32)
            
            batch_images.append(img)
            batch_labels.append(labels)
        
        # Pad batch if necessary
        while len(batch_images) < self.batch_size:
            batch_images.append(np.zeros((*self.image_size, 3), dtype=np.float32))
            batch_labels.append(np.zeros(len(self.disease_labels), dtype=np.float32))
        
        return np.array(batch_images), np.array(batch_labels)