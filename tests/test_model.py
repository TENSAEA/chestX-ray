#!/usr/bin/env python3
"""
Unit tests for the chest X-ray AI model
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import shutil
from unittest.mock import Mock, patch
import tensorflow as tf

# Import modules to test
import sys
sys.path.append('..')
from data_generator import XrayDataGenerator
from model_builder import ModelBuilder
from advanced_features import ModelEvaluator, UncertaintyEstimator

class TestDataGenerator(unittest.TestCase):
    """Test cases for XrayDataGenerator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.temp_dir, 'images')
        os.makedirs(self.images_dir)
        
        # Create sample CSV data
        self.sample_data = pd.DataFrame({
            'Image Index': ['test1.png', 'test2.png', 'test3.png'],
            'Atelectasis': [1, 0, 1],
            'Cardiomegaly': [0, 1, 0],
            'Effusion': [1, 1, 0]
        })
        
        self.disease_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion']
        
        # Create dummy images
        for img_name in self.sample_data['Image Index']:
            img_path = os.path.join(self.images_dir, img_name)
            # Create a dummy image file
            dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            tf.keras.preprocessing.image.save_img(img_path, dummy_img)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_data_generator_initialization(self):
        """Test data generator initialization"""
        generator = XrayDataGenerator(
            images_dir=self.images_dir,
            batch_size=2,
            image_size=(224, 224)
        )
        
        self.assertEqual(generator.batch_size, 2)
        self.assertEqual(generator.image_size, (224, 224))
        self.assertEqual(generator.images_dir, self.images_dir)
    
    def test_load_and_preprocess_image(self):
        """Test image loading and preprocessing"""
        generator = XrayDataGenerator(
            images_dir=self.images_dir,
            batch_size=2,
            image_size=(224, 224)
        )
        
        img_path = os.path.join(self.images_dir, 'test1.png')
        img = generator.load_and_preprocess_image(img_path)
        
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (224, 224, 3))
        self.assertTrue(np.all(img >= 0) and np.all(img <= 255))
    
    def test_create_generator(self):
        """Test generator creation"""
        generator = XrayDataGenerator(
            images_dir=self.images_dir,
            batch_size=2,
            image_size=(224, 224)
        )
        
        data_gen = generator.create_generator(
            self.sample_data, 
            self.disease_labels, 
            shuffle=False
        )
        
        # Test one batch
        batch_x, batch_y = next(data_gen)
        
        self.assertEqual(batch_x.shape[0], 2)  # Batch size
        self.assertEqual(batch_x.shape[1:], (224, 224, 3))  # Image shape
        self.assertEqual(batch_y.shape, (2, 3))  # Labels shape

class TestModelBuilder(unittest.TestCase):
    """Test cases for ModelBuilder"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'image_size': [224, 224],
            'dropout_rate': 0.5,
            'disease_labels': ['Atelectasis', 'Cardiomegaly', 'Effusion']
        }
    
    def test_model_builder_initialization(self):
        """Test model builder initialization"""
        builder = ModelBuilder(self.config)
        
        self.assertEqual(builder.image_size, (224, 224))
        self.assertEqual(builder.num_classes, 3)
        self.assertEqual(builder.dropout_rate, 0.5)
    
    def test_build_model(self):
        """Test model building"""
        builder = ModelBuilder(self.config)
        model = builder.build_model()
        
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        self.assertEqual(model.output_shape, (None, 3))
    
    def test_model_compilation(self):
        """Test model compilation"""
        builder = ModelBuilder(self.config)
        model = builder.build_model()
        
        # Check if model is compiled
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)

class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple mock model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(3, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        
        self.disease_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion']
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        evaluator = ModelEvaluator(self.model, self.disease_labels)
        
        self.assertEqual(evaluator.model, self.model)
        self.assertEqual(evaluator.disease_labels, self.disease_labels)
    
    def test_calculate_metrics(self):
        """Test metrics calculation"""
        evaluator = ModelEvaluator(self.model, self.disease_labels)
        
        # Create dummy predictions
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_pred = np.array([[0.8, 0.2, 0.9], [0.1, 0.7, 0.3], [0.6, 0.8, 0.2]])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        self.assertIn('mean_auc', metrics)
        self.assertIn('disease_aucs', metrics)
        self.assertTrue(0 <= metrics['mean_auc'] <= 1)

class TestUncertaintyEstimator(unittest.TestCase):
    """Test cases for UncertaintyEstimator"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a model with dropout for uncertainty estimation
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
    
    def test_uncertainty_estimator_initialization(self):
        """Test uncertainty estimator initialization"""
        estimator = UncertaintyEstimator(self.model)
        
        self.assertEqual(estimator.model, self.model)
        self.assertEqual(estimator.num_samples, 100)  # Default value
    
    def test_monte_carlo_prediction(self):
        """Test Monte Carlo prediction"""
        estimator = UncertaintyEstimator(self.model, num_samples=10)
        
        # Create dummy input
        x = np.random.random((1, 224, 224, 3))
        
        predictions, uncertainty = estimator.monte_carlo_prediction(x)
        
        self.assertEqual(predictions.shape, (1, 3))
        self.assertEqual(uncertainty.shape, (1, 3))
        self.assertTrue(np.all(uncertainty >= 0))

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.temp_dir, 'images')
        os.makedirs(self.images_dir)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'Image Index': ['test1.png', 'test2.png', 'test3.png', 'test4.png'],
            'Atelectasis': [1, 0, 1, 0],
            'Cardiomegaly': [0, 1, 0, 1],
            'Effusion': [1, 1, 0, 0]
        })
        
        self.disease_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion']
        
        # Create dummy images
        for img_name in self.sample_data['Image Index']:
            img_path = os.path.join(self.images_dir, img_name)
            dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            tf.keras.preprocessing.image.save_img(img_path, dummy_img)
        
        # Create config
        self.config = {
            'image_size': [224, 224],
            'dropout_rate': 0.5,
            'disease_labels': self.disease_labels,
            'batch_size': 2
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data loading to prediction"""
        # 1. Create data generator
        data_generator = XrayDataGenerator(
            images_dir=self.images_dir,
            batch_size=self.config['batch_size'],
            image_size=tuple(self.config['image_size'])
        )
        
        # 2. Build model
        model_builder = ModelBuilder(self.config)
        model = model_builder.build_model()
        
        # 3. Create data generator
        train_generator = data_generator.create_generator(
            self.sample_data, 
            self.disease_labels, 
            shuffle=True
        )
        
        # 4. Train for one step (just to test the pipeline)
        batch_x, batch_y = next(train_generator)
        loss = model.train_on_batch(batch_x, batch_y)
        
        # 5. Make predictions
        predictions = model.predict(batch_x)
        
        # Assertions
        self.assertIsInstance(loss, (float, np.float32, np.float64))
        self.assertEqual(predictions.shape, (2, 3))
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1))
    
    def test_evaluation_pipeline(self):
        """Test evaluation pipeline"""
        # Create and train a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(3, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Create evaluator
        evaluator = ModelEvaluator(model, self.disease_labels)
        
        # Create dummy predictions
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[0.8, 0.2, 0.9], [0.1, 0.7, 0.3]])
        
        # Test evaluation
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('mean_auc', metrics)
        self.assertIn('disease_aucs', metrics)

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_missing_image_file(self):
        """Test handling of missing image files"""
        temp_dir = tempfile.mkdtemp()
        images_dir = os.path.join(temp_dir, 'images')
        os.makedirs(images_dir)
        
        try:
            generator = XrayDataGenerator(
                images_dir=images_dir,
                batch_size=1,
                image_size=(224, 224)
            )
            
            # Try to load non-existent image
            img = generator.load_and_preprocess_image('nonexistent.png')
            self.assertIsNone(img)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_invalid_image_format(self):
        """Test handling of invalid image formats"""
        temp_dir = tempfile.mkdtemp()
        images_dir = os.path.join(temp_dir, 'images')
        os.makedirs(images_dir)
        
        try:
            # Create invalid image file
            invalid_img_path = os.path.join(images_dir, 'invalid.png')
            with open(invalid_img_path, 'w') as f:
                f.write('This is not an image')
            
            generator = XrayDataGenerator(
                images_dir=images_dir,
                batch_size=1,
                image_size=(224, 224)
            )
            
            # Try to load invalid image
            img = generator.load_and_preprocess_image(invalid_img_path)
            self.assertIsNone(img)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_empty_dataset(self):
        """Test handling of empty datasets"""
        temp_dir = tempfile.mkdtemp()
        images_dir = os.path.join(temp_dir, 'images')
        os.makedirs(images_dir)
        
        try:
            generator = XrayDataGenerator(
                images_dir=images_dir,
                batch_size=1,
                image_size=(224, 224)
            )
            
            empty_df = pd.DataFrame(columns=['Image Index', 'Atelectasis'])
            disease_labels = ['Atelectasis']
            
            # This should handle empty dataset gracefully
            data_gen = generator.create_generator(empty_df, disease_labels)
            
            # Should not raise an exception
            self.assertIsNotNone(data_gen)
            
        finally:
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    # Set up TensorFlow for testing
    tf.config.experimental.set_memory_growth(
        tf.config.experimental.list_physical_devices('GPU')[0], True
    ) if tf.config.experimental.list_physical_devices('GPU') else None
    
    # Run tests
    unittest.main(verbosity=2)