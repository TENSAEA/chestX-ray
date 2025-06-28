import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image, ImageDraw, ImageFilter
import random

class MockDataGenerator:
    def __init__(self, num_samples=1000, image_size=(224, 224)):
        self.num_samples = num_samples
        self.image_size = image_size
        self.disease_labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        
    def generate_mock_xray_image(self, diseases, image_id):
        """Generate a realistic-looking mock chest X-ray image"""
        # Create base chest X-ray shape
        img = Image.new('L', self.image_size, color=20)  # Dark background
        draw = ImageDraw.Draw(img)
        
        # Draw chest cavity outline
        chest_width = int(self.image_size[0] * 0.7)
        chest_height = int(self.image_size[1] * 0.8)
        chest_x = (self.image_size[0] - chest_width) // 2
        chest_y = int(self.image_size[1] * 0.1)
        
        # Draw ribcage structure
        for i in range(8):
            y_pos = chest_y + (i * chest_height // 10)
            # Left ribs
            draw.arc([chest_x, y_pos, chest_x + chest_width//2, y_pos + 30], 
                    start=0, end=180, fill=80, width=2)
            # Right ribs
            draw.arc([chest_x + chest_width//2, y_pos, chest_x + chest_width, y_pos + 30], 
                    start=0, end=180, fill=80, width=2)
        
        # Draw spine
        spine_x = self.image_size[0] // 2
        draw.line([spine_x, chest_y, spine_x, chest_y + chest_height], fill=90, width=3)
        
        # Draw heart shadow
        heart_x = spine_x - 40
        heart_y = chest_y + chest_height // 3
        draw.ellipse([heart_x, heart_y, heart_x + 80, heart_y + 100], fill=60)
        
        # Add lung fields
        # Left lung
        draw.ellipse([chest_x + 20, chest_y + 20, spine_x - 10, chest_y + chest_height - 20], fill=40)
        # Right lung
        draw.ellipse([spine_x + 10, chest_y + 20, chest_x + chest_width - 20, chest_y + chest_height - 20], fill=40)
        
        # Add disease-specific patterns
        if 'Cardiomegaly' in diseases:
            # Enlarged heart
            draw.ellipse([heart_x - 20, heart_y - 10, heart_x + 100, heart_y + 120], fill=50)
            
        if 'Pneumonia' in diseases:
            # Add cloudy patches
            for _ in range(random.randint(2, 5)):
                x = random.randint(chest_x + 20, chest_x + chest_width - 40)
                y = random.randint(chest_y + 20, chest_y + chest_height - 40)
                draw.ellipse([x, y, x + 30, y + 30], fill=70)
                
        if 'Pneumothorax' in diseases:
            # Add dark areas (collapsed lung)
            side = random.choice(['left', 'right'])
            if side == 'left':
                draw.rectangle([chest_x + 20, chest_y + 20, spine_x - 10, chest_y + 60], fill=25)
            else:
                draw.rectangle([spine_x + 10, chest_y + 20, chest_x + chest_width - 20, chest_y + 60], fill=25)
                
        if 'Effusion' in diseases:
            # Add fluid at bottom of lungs
            draw.rectangle([chest_x + 20, chest_y + chest_height - 60, 
                          chest_x + chest_width - 20, chest_y + chest_height - 20], fill=55)
            
        if 'Mass' in diseases or 'Nodule' in diseases:
            # Add circular masses
            for _ in range(random.randint(1, 3)):
                x = random.randint(chest_x + 30, chest_x + chest_width - 50)
                y = random.randint(chest_y + 30, chest_y + chest_height - 50)
                size = random.randint(15, 35) if 'Mass' in diseases else random.randint(5, 15)
                draw.ellipse([x, y, x + size, y + size], fill=85)
        
        # Add noise and texture
        img_array = np.array(img)
        noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
        img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Apply slight blur for realism
        img = Image.fromarray(img_array)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return img
    
    def generate_mock_dataset(self, output_dir='mock_images'):
        """Generate complete mock dataset"""
        os.makedirs(output_dir, exist_ok=True)
        
        data_records = []
        
        for i in range(self.num_samples):
            image_id = f"mock_xray_{i:05d}.png"
            
            # Randomly assign diseases (some images have no findings)
            num_diseases = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])  # Most have 0-1 diseases
            
            if num_diseases == 0:
                diseases = []
            else:
                diseases = random.sample(self.disease_labels, num_diseases)
            
            # Generate image
            img = self.generate_mock_xray_image(diseases, image_id)
            img_path = os.path.join(output_dir, image_id)
            img.save(img_path)
            
            # Create record
            record = {
                'Image Index': image_id,
                'Patient ID': f"P{i:05d}",
                'Patient Age': random.randint(20, 80),
                'Patient Gender': random.choice(['M', 'F']),
                'View Position': random.choice(['PA', 'AP']),
                'Finding Labels': '|'.join(diseases) if diseases else 'No Finding'
            }
            
            # Add binary columns for each disease
            for disease in self.disease_labels:
                record[disease] = 1 if disease in diseases else 0
            
            data_records.append(record)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{self.num_samples} images")
        
        # Create DataFrame and save
        df = pd.DataFrame(data_records)
        df.to_csv(os.path.join('data', 'mock_chest_xray_data.csv'), index=False)
        
        print(f"Mock dataset generated successfully!")
        print(f"Images saved to: {output_dir}")
        print(f"CSV saved to: data/mock_chest_xray_data.csv")
        print(f"Dataset shape: {df.shape}")
        
        return df

# Generate mock dataset
if __name__ == "__main__":
    generator = MockDataGenerator(num_samples=500)  # Smaller for quick training
    df = generator.generate_mock_dataset()
    print("\nDataset summary:")
    print(df['Finding Labels'].value_counts().head(10))