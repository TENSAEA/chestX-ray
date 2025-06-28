import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

class ChestXrayModel:
    def __init__(self, num_classes=14, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        
    def build_model(self):
        """Build the chest X-ray classification model"""
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu', name='dense_1')(x)
        x = Dropout(0.5, name='dropout_1')(x)
        x = Dense(256, activation='relu', name='dense_2')(x)
        x = Dropout(0.3, name='dropout_2')(x)
        
        # Output layer for multi-label classification
        predictions = Dense(self.num_classes, activation='sigmoid', name='predictions')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['AUC', 'binary_accuracy']
        )
        
        return model
    
    def get_callbacks(self, model_path='best_model.h5'):
        """Get training callbacks"""
        callbacks = [
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        return callbacks
    
    def unfreeze_top_layers(self, model, num_layers=50):
        """Unfreeze top layers for fine-tuning"""
        base_model = model.layers[0]
        base_model.trainable = True
        
        # Freeze all layers except the top num_layers
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['AUC', 'binary_accuracy']
        )
        
        return model