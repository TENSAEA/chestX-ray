import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Handle optional imports gracefully
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    st.warning("OpenCV not available. Some image processing features may be limited.")

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
    # Test if TensorFlow is working
    try:
        tf.constant([1, 2, 3])
    except Exception as e:
        HAS_TENSORFLOW = False
        st.error(f"TensorFlow import failed: {e}")
except ImportError:
    HAS_TENSORFLOW = False

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

# Set page config
st.set_page_config(
    page_title="🫁 Chest X-ray AI Diagnostics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-card {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-card {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
    }
</style>
""", unsafe_allow_html=True)

# Mock model class for demonstration
class MockChestXrayModel:
    def __init__(self):
        self.disease_labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        self.is_loaded = True
    
    def predict(self, image_array):
        """Mock prediction function"""
        np.random.seed(42)  # For consistent demo results
        predictions = np.random.beta(2, 5, len(self.disease_labels))
        uncertainties = np.random.beta(2, 8, len(self.disease_labels)) * 0.3
        return predictions, uncertainties

@st.cache_resource
def load_model_and_data():
    """Load the trained model and associated data"""
    try:
        if HAS_TENSORFLOW:
            # In a real scenario, you would load your actual model here
            # model = tf.keras.models.load_model('path_to_your_model.h5')
            st.info("🔄 TensorFlow available - using mock model for demonstration")
        else:
            st.warning("⚠️ TensorFlow not available - using mock model for demonstration")
        
        # Use mock model for demonstration
        model = MockChestXrayModel()
        
        disease_labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        
        return model, disease_labels, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, False

def preprocess_image(image):
    """Preprocess uploaded image"""
    try:
        # Convert PIL to numpy array
        img_array = np.array(image.convert('RGB'))
        
        # Resize image
        if HAS_OPENCV:
            img_resized = cv2.resize(img_array, (224, 224))
        else:
            # Fallback using PIL
            img_pil = image.resize((224, 224))
            img_resized = np.array(img_pil.convert('RGB'))
        
        # Normalize
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, img_resized
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None

def predict_with_uncertainty(model, image, n_samples=10):
    """Predict with uncertainty estimation"""
    try:
        if hasattr(model, 'predict'):
            predictions, uncertainties = model.predict(image)
        else:
            # Fallback for mock model
            np.random.seed(42)
            predictions = np.random.beta(2, 5, 14)
            uncertainties = np.random.beta(2, 8, 14) * 0.3
        
        return predictions, uncertainties
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        # Return dummy data
        return np.random.random(14), np.random.random(14) * 0.1

def generate_gradcam_heatmap(image, class_index):
    """Generate a simple heatmap for visualization"""
    try:
        if HAS_OPENCV:
            img_gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Create a heatmap based on image gradients (mock Grad-CAM)
            grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            heatmap = np.sqrt(grad_x**2 + grad_y**2)
            
            # Normalize and apply some randomness based on class_index
            np.random.seed(class_index)
            noise = np.random.normal(0, 0.1, heatmap.shape)
            heatmap = heatmap + noise
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
            # Apply Gaussian blur for smoother heatmap
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        else:
            # Fallback without OpenCV
            img_gray = np.mean(image, axis=2)
            np.random.seed(class_index)
            heatmap = np.random.random(img_gray.shape)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap
    except Exception as e:
        st.error(f"Error generating heatmap: {e}")
        return np.random.random((224, 224))

def create_prediction_visualization(predictions, uncertainties, disease_labels):
    """Create beautiful prediction visualization"""
    try:
        # Combine predictions and uncertainties
        df = pd.DataFrame({
            'Disease': disease_labels,
            'Probability': predictions,
            'Uncertainty': uncertainties,
            'Confidence': 1 - uncertainties
        })
        
        # Sort by probability
        df = df.sort_values('Probability', ascending=True)
        
        # Create horizontal bar chart with uncertainty
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            y=df['Disease'],
            x=df['Probability'],
            orientation='h',
            marker=dict(
                color=df['Probability'],
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Probability")
            ),
            text=[f'{p:.3f} ± {u:.3f}' for p, u in zip(df['Probability'], df['Uncertainty'])],
            textposition='auto',
            name='Predictions'
        ))
        
        fig.update_layout(
            title='Disease Prediction Probabilities with Uncertainty',
            xaxis_title='Probability',
            yaxis_title='Disease',
            height=600,
            template='plotly_white',
            font=dict(size=12)
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return go.Figure()

def create_uncertainty_gauge(avg_uncertainty):
    """Create uncertainty gauge"""
    try:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = avg_uncertainty,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Model Uncertainty"},
            delta = {'reference': 0.1},
            gauge = {
                'axis': {'range': [None, 0.5]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.1], 'color': "lightgreen"},
                    {'range': [0.1, 0.2], 'color': "yellow"},
                    {'range': [0.2, 0.5], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.2
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    except Exception as e:
        st.error(f"Error creating gauge: {e}")
        return go.Figure()

def main():
    # Header
    st.markdown('<h1 class="main-header">🫁 Chest X-ray AI Diagnostics</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d;">Advanced Deep Learning System for Chest X-ray Disease Detection</p>', unsafe_allow_html=True)
    
    # System status
    col1, col2, col3 = st.columns(3)
    with col1:
        if HAS_TENSORFLOW:
            st.success("✅ TensorFlow: Available")
        else:
            st.error("❌ TensorFlow: Not Available")
    
    with col2:
        if HAS_OPENCV:
            st.success("✅ OpenCV: Available")
        else:
            st.warning("⚠️ OpenCV: Limited")
    
    with col3:
        if HAS_SKLEARN:
            st.success("✅ Scikit-learn: Available")
        else:
            st.warning("⚠️ Scikit-learn: Not Available")
    
    # Load model
    model, disease_labels, model_loaded = load_model_and_data()
    
    if not model_loaded:
        st.error("❌ Failed to load model. Using demonstration mode.")
        model = MockChestXrayModel()
        disease_labels = model.disease_labels
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🔬 Navigation")
        
        if HAS_OPTION_MENU:
            selected = option_menu(
                menu_title=None,
                options=["🏠 Home", "📊 Predict", "📈 Analytics", "ℹ️ About"],
                icons=["house", "upload", "graph-up", "info-circle"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                    "icon": {"color": "#3498db", "font-size": "18px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                    "nav-link-selected": {"background-color": "#3498db"},
                }
            )
        else:
            # Fallback navigation without option_menu
            selected = st.selectbox(
                "Choose a page:",
                ["🏠 Home", "📊 Predict", "📈 Analytics", "ℹ️ About"]
            )
    
    # Main content based on selection
    if selected == "🏠 Home":
        show_home_page()
    elif selected == "📊 Predict":
        show_prediction_page(model, disease_labels)
    elif selected == "📈 Analytics":
        show_analytics_page()
    elif selected == "ℹ️ About":
        show_about_page()

def show_home_page():
    """Show home page with overview"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 14 Diseases</h3>
            <p>Detects multiple chest conditions simultaneously</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 Deep Learning</h3>
            <p>Advanced AI architecture with uncertainty estimation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Real-time</h3>
            <p>Instant analysis with confidence scores</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">🔍 System Features</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚀 Advanced AI Capabilities
        - **Multi-label Classification**: Detects 14 different chest conditions
        - **Uncertainty Estimation**: Provides confidence assessment
        - **Visual Explanations**: Shows which areas influenced decisions
        - **Real-time Processing**: Instant results for uploaded images
        """)
        
        st.markdown("""
        ### 🎯 Detected Conditions
        - Atelectasis
        - Cardiomegaly
        - Effusion
        - Infiltration
        - Mass
        - Nodule
        - Pneumonia
        - Pneumothorax
        - Consolidation
        - Edema
        - Emphysema
        - Fibrosis
        - Pleural Thickening
        - Hernia
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Demo Performance Metrics
        Our system demonstrates excellent performance across multiple diseases:
        """)
        
        # Create a sample performance chart
        diseases = ['Pneumonia', 'Cardiomegaly', 'Effusion', 'Mass', 'Nodule']
        aucs = [0.89, 0.85, 0.82, 0.78, 0.75]
        
        try:
            fig = px.bar(
                x=aucs, 
                y=diseases, 
                orientation='h',
                title='Sample AUC Scores by Disease',
                color=aucs,
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating chart: {e}")
        
        st.markdown("""
        ### ⚡ Quick Start
        1. Navigate to the **Predict** tab
        2. Upload a chest X-ray image
        3. Get instant AI diagnosis with confidence scores
        4. View visualization of decision areas
        """)
        
        system_info = {
            'Component': ['Python Version', 'TensorFlow', 'NumPy', 'Pandas', 'OpenCV', 'Plotly'],
            'Status': [
                f"✅ {'.'.join(map(str, [3, 11]))}" if True else "❌ Not Available",
                "✅ Available" if HAS_TENSORFLOW else "❌ Not Available",
                f"✅ {np.__version__}",
                f"✅ {pd.__version__}",
                "✅ Available" if HAS_OPENCV else "❌ Not Available",
                "✅ Available"
            ],
            'Purpose': [
                'Runtime Environment',
                'Deep Learning Framework',
                'Numerical Computing',
                'Data Manipulation',
                'Image Processing',
                'Interactive Visualizations'
            ]
        }
        
        df_system = pd.DataFrame(system_info)
        st.dataframe(df_system, use_container_width=True)
        
        # Model architecture info
        st.markdown("### 🏗️ Model Architecture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Base Architecture:**
            - ResNet50 backbone
            - Pre-trained on ImageNet
            - Fine-tuned for chest X-rays
            - Multi-label classification head
            
            **Input Specifications:**
            - Image Size: 224 × 224 pixels
            - Color Channels: 3 (RGB)
            - Normalization: [0, 1] range
            - Batch Size: Flexible
            """)
        
        with col2:
            st.markdown("""
            **Training Details:**
            - Optimizer: Adam
            - Learning Rate: 0.001 → 0.0001
            - Loss Function: Binary Cross-entropy
            - Regularization: Dropout (0.5)
            
            **Data Augmentation:**
            - Random Rotation: ±15°
            - Random Scaling: 0.9-1.1×
            - Brightness Adjustment: ±20%
            - Horizontal Flipping: 50%
            """)
    
    with tab4:
        st.markdown("### 📉 Demo Training History")
        
        # Mock training history
        epochs = list(range(1, 21))
        train_loss = [0.8 - 0.03*i + 0.01*np.sin(i) + np.random.normal(0, 0.01) for i in epochs]
        val_loss = [0.85 - 0.025*i + 0.015*np.sin(i) + np.random.normal(0, 0.015) for i in epochs]
        train_auc = [0.6 + 0.015*i - 0.005*np.sin(i) + np.random.normal(0, 0.01) for i in epochs]
        val_auc = [0.58 + 0.014*i - 0.007*np.sin(i) + np.random.normal(0, 0.015) for i in epochs]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss curves
            try:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='Training Loss',
                                       line=dict(color='blue', width=2)))
                fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Validation Loss',
                                       line=dict(color='red', width=2)))
                fig.update_layout(title='Training & Validation Loss', 
                                xaxis_title='Epoch', yaxis_title='Loss',
                                height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating loss chart: {e}")
        
        with col2:
            # AUC curves
            try:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=epochs, y=train_auc, name='Training AUC',
                                       line=dict(color='green', width=2)))
                fig.add_trace(go.Scatter(x=epochs, y=val_auc, name='Validation AUC',
                                       line=dict(color='orange', width=2)))
                fig.update_layout(title='Training & Validation AUC', 
                                xaxis_title='Epoch', yaxis_title='AUC',
                                height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating AUC chart: {e}")
        
        # Training summary
        st.markdown("### 📋 Training Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final Train Loss", f"{train_loss[-1]:.4f}")
        with col2:
            st.metric("Final Val Loss", f"{val_loss[-1]:.4f}")
        with col3:
            st.metric("Final Train AUC", f"{train_auc[-1]:.4f}")
        with col4:
            st.metric("Final Val AUC", f"{val_auc[-1]:.4f}")

def show_about_page():
    """Show about page with system information"""
    st.markdown('<h2 class="sub-header">ℹ️ About This System</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        
        This Chest X-ray AI Diagnostics system represents a demonstration of automated 
        medical image analysis using deep learning techniques. It showcases the potential 
        for AI to assist in detecting multiple chest conditions from X-ray images.
        
        ### 🔬 Technical Architecture
        
        **Model Architecture:**
        - **Base Model**: ResNet50 pre-trained on ImageNet
        - **Transfer Learning**: Fine-tuned on chest X-ray data
        - **Multi-label Classification**: Sigmoid activation for independent disease prediction
        - **Uncertainty Estimation**: Monte Carlo Dropout for confidence assessment
        
        **Key Features:**
        - **Real-time Processing**: Instant analysis of uploaded images
        - **Multi-disease Detection**: Simultaneous screening for 14 conditions
        - **Confidence Scoring**: Uncertainty quantification for each prediction
        - **Visual Explanations**: Heatmaps showing decision-relevant regions
        
        ### 📊 Performance Highlights
        
        - **Mean AUC**: 0.847 across all 14 diseases (demonstration values)
        - **Best Individual Performance**: Pneumonia (AUC: 0.89)
        - **Processing Speed**: < 2 seconds per image
        - **Model Size**: 94MB (optimized for deployment)
        
        ### 🛡️ Important Limitations
        
        **This is a demonstration system:**
        - **Not for clinical use**: Educational and research purposes only
        - **Mock predictions**: Uses simulated model outputs for demonstration
        - **Requires validation**: Real clinical deployment needs extensive testing
        - **Expert oversight**: Always requires medical professional review
        
        ### 🔧 Technical Implementation
        
        **Framework & Libraries:**
        - **Core**: Python, NumPy, Pandas
        - **ML Framework**: TensorFlow/Keras (when available)
        - **Image Processing**: OpenCV, PIL
        - **Visualization**: Plotly, Matplotlib, Seaborn
        - **Web Interface**: Streamlit
        
        **Deployment Considerations:**
        - **Cloud Compatibility**: Optimized for Streamlit Cloud
        - **Dependency Management**: Graceful handling of missing packages
        - **Error Handling**: Robust error management and user feedback
        - **Performance**: Efficient processing for web deployment
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Demo Statistics
        """)
        
        # Model info cards
        stats = [
            ("🎯", "Diseases Detected", "14"),
            ("📸", "Demo Images", "Unlimited"),
            ("🧠", "Model Parameters", "23.5M"),
            ("⚡", "Processing Time", "<2s"),
            ("📊", "Demo AUC", "0.847"),
            ("🎓", "Training Epochs", "50"),
            ("💾", "Model Size", "94MB"),
            ("🔄", "Augmentations", "8 Types")
        ]
        
        for icon, label, value in stats:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; margin: 0.5rem 0; 
                       border-radius: 8px; border-left: 4px solid #3498db;
                       box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                    <div>
                        <div style="font-weight: bold; color: #2c3e50;">{value}</div>
                        <div style="font-size: 0.9rem; color: #7f8c8d;">{label}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🏆 Key Features
        
        ✅ **Multi-label Detection**  
        ✅ **Uncertainty Estimation**  
        ✅ **Visual Explanations**  
        ✅ **Real-time Processing**  
        ✅ **Interactive Interface**  
        ✅ **Error Handling**  
        ✅ **Cloud Deployment**  
        ✅ **Educational Focus**  
        """)
    
    # Dataset information
    st.markdown('<h3 class="sub-header">📚 Dataset Information</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 📊 NIH Chest X-ray Dataset
        - **Total Images**: 112,120
        - **Unique Patients**: 30,805
        - **Image Size**: 1024 × 1024 pixels
        - **Format**: PNG, 8-bit grayscale
        - **View Position**: PA (Posterior-Anterior)
        """)
    
    with col2:
        st.markdown("""
        #### 🏥 Disease Categories
        - **Lung Diseases**: Pneumonia, Atelectasis, Pneumothorax
        - **Heart Conditions**: Cardiomegaly
        - **Fluid-related**: Effusion, Edema
        - **Tissue Changes**: Fibrosis, Emphysema
        - **Masses**: Mass, Nodule
        - **Other**: Infiltration, Consolidation, etc.
        """)
    
    with col3:
        st.markdown("""
        #### 📈 Data Distribution
        - **Training Set**: 70% (78,484 images)
        - **Validation Set**: 15% (16,818 images)
        - **Test Set**: 15% (16,818 images)
        - **Label Distribution**: Highly imbalanced
        - **Multi-label**: 60% have multiple conditions
        """)
    
    # System Status
    st.markdown('<h3 class="sub-header">🖥️ System Status</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Core Dependencies:**
        """)
        deps = [
            ("Python", "✅ Available"),
            ("NumPy", "✅ Available"),
            ("Pandas", "✅ Available"),
            ("Matplotlib", "✅ Available"),
            ("Plotly", "✅ Available")
        ]
        
        for dep, status in deps:
            st.markdown(f"- **{dep}**: {status}")
    
    with col2:
        st.markdown("""
        **Optional Dependencies:**
        """)
        opt_deps = [
            ("TensorFlow", "✅ Available" if HAS_TENSORFLOW else "❌ Not Available"),
            ("OpenCV", "✅ Available" if HAS_OPENCV else "❌ Not Available"),
            ("Scikit-learn", "✅ Available" if HAS_SKLEARN else "❌ Not Available"),
            ("Option Menu", "✅ Available" if HAS_OPTION_MENU else "❌ Not Available")
        ]
        
        for dep, status in opt_deps:
            st.markdown(f"- **{dep}**: {status}")
    
    with col3:
        st.markdown("""
        **System Info:**
        """)
        st.markdown(f"""
        - **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - **Mode**: {'Production' if HAS_TENSORFLOW else 'Demo'}
        - **Platform**: Streamlit Cloud
        - **Status**: 🟢 Operational
        """)
    
    # Medical disclaimer
    st.markdown("---")
    st.markdown("""
    <div class="warning-card">
        <h4 style="color: #856404; margin-top: 0;">⚠️ Important Medical Disclaimer</h4>
        <p style="color: #856404; margin-bottom: 0;">
            <strong>This system is for educational and demonstration purposes only.</strong><br><br>
            
            • Not intended for clinical diagnosis or treatment decisions<br>
            • Should not replace professional medical consultation<br>
            • Always consult qualified healthcare professionals<br>
            • Results may not be accurate for all patient populations<br>
            • System performance may vary with image quality and patient factors<br>
            • This is a demonstration using simulated model outputs
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Version and contact info
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Version Information:**  
        🔢 **Demo Version**: v1.0.0  
        📅 **Last Updated**: December 2024  
        🔄 **Update Type**: Educational Demo  
        """)
    
    with col2:
        st.markdown("""
        **System Status:**  
        🟢 **Demo Status**: Active  
        🟢 **UI Status**: Stable  
        🟡 **ML Status**: Simulated  
        """)
    
    with col3:
        st.markdown("""
        **Usage Stats:**  
        ⚡ **Response Time**: <2s  
        📊 **Accuracy**: Demo Mode  
        👥 **Purpose**: Educational  
        """)

# Error handling wrapper
def safe_main():
    """Main function with error handling"""
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.markdown("""
        <div class="error-card">
            <h4>🚨 System Error</h4>
            <p>The application encountered an unexpected error. This might be
            due to missing dependencies or system limitations.</p>
            <p><strong>Troubleshooting steps:</strong></p>
            <ul>
                <li>Refresh the page and try again</li>
                <li>Check if all required files are uploaded correctly</li>
                <li>Ensure image format is supported (PNG, JPG, JPEG)</li>
                <li>Try with a smaller image file</li>
            </ul>
            <p>If the problem persists, this may be due to missing system dependencies
            in the deployment environment.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show system diagnostic info
        st.markdown("### 🔧 System Diagnostic Information")
        
        diagnostic_info = {
            'Component': ['TensorFlow', 'OpenCV', 'Scikit-learn', 'Option Menu', 'NumPy', 'Pandas'],
            'Status': [
                '✅ Available' if HAS_TENSORFLOW else '❌ Missing',
                '✅ Available' if HAS_OPENCV else '❌ Missing', 
                '✅ Available' if HAS_SKLEARN else '❌ Missing',
                '✅ Available' if HAS_OPTION_MENU else '❌ Missing',
                f'✅ {np.__version__}',
                f'✅ {pd.__version__}'
            ],
            'Impact': [
                'Core ML functionality' if not HAS_TENSORFLOW else 'None',
                'Advanced image processing' if not HAS_OPENCV else 'None',
                'Additional ML metrics' if not HAS_SKLEARN else 'None', 
                'Enhanced navigation UI' if not HAS_OPTION_MENU else 'None',
                'None',
                'None'
            ]
        }
        
        df_diagnostic = pd.DataFrame(diagnostic_info)
        st.dataframe(df_diagnostic, use_container_width=True)

# Additional utility functions for robustness
def create_fallback_chart(title="Chart", message="Chart unavailable"):
    """Create a fallback chart when plotting fails"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        height=400
    )
    return fig

def validate_image(image):
    """Validate uploaded image"""
    try:
        # Check if image can be opened
        if image is None:
            return False, "No image provided"
        
        # Check image format
        if image.format not in ['PNG', 'JPEG', 'JPG']:
            return False, f"Unsupported format: {image.format}. Please use PNG, JPG, or JPEG."
        
        # Check image size
        width, height = image.size
        if width < 50 or height < 50:
            return False, "Image too small. Minimum size is 50x50 pixels."
        
        if width > 5000 or height > 5000:
            return False, "Image too large. Maximum size is 5000x5000 pixels."
        
        # Check file size (approximate)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_size_mb = len(img_byte_arr.getvalue()) / (1024 * 1024)
        
        if img_size_mb > 10:
            return False, "Image file too large. Maximum size is 10MB."
        
        return True, "Image is valid"
    
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

def create_sample_predictions():
    """Create sample predictions for demonstration"""
    np.random.seed(42)  # For consistent results
    
    disease_labels = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 
        'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    
    # Create realistic-looking predictions (mostly low probabilities with a few higher ones)
    predictions = np.random.beta(1.5, 8, len(disease_labels))  # Skewed towards 0
    
    # Make a few diseases more likely
    high_prob_indices = np.random.choice(len(disease_labels), 2, replace=False)
    for idx in high_prob_indices:
        predictions[idx] = np.random.beta(3, 2)  # Higher probability
    
    # Create corresponding uncertainties (lower for higher predictions)
    uncertainties = np.random.beta(2, 5, len(disease_labels)) * 0.3
    for idx in high_prob_indices:
        uncertainties[idx] *= 0.5  # Lower uncertainty for high-confidence predictions
    
    return predictions, uncertainties

def format_prediction_summary(predictions, uncertainties, disease_labels, top_n=3):
    """Format a summary of top predictions"""
    # Get top predictions
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    
    summary = "### 🎯 Top Predictions:\n\n"
    
    for i, idx in enumerate(top_indices):
        disease = disease_labels[idx]
        prob = predictions[idx]
        uncertainty = uncertainties[idx]
        
        # Determine confidence level
        if uncertainty < 0.1:
            confidence = "High"
            icon = "🟢"
        elif uncertainty < 0.2:
            confidence = "Medium" 
            icon = "🟡"
        else:
            confidence = "Low"
            icon = "🔴"
        
        summary += f"{i+1}. **{disease}**: {prob:.1%} ± {uncertainty:.1%} {icon} ({confidence} confidence)\n"
    
    return summary

def create_system_info_card():
    """Create system information card"""
    return f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0;">
        <h3 style="margin-top: 0; color: white;">🖥️ System Information</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div>
                <strong>Core Status:</strong><br>
                • Python: ✅ Active<br>
                • NumPy: ✅ v{np.__version__}<br>
                • Pandas: ✅ v{pd.__version__}<br>
                • Plotly: ✅ Active
            </div>
            <div>
                <strong>Optional Components:</strong><br>
                • TensorFlow: {'✅ Available' if HAS_TENSORFLOW else '❌ Missing'}<br>
                • OpenCV: {'✅ Available' if HAS_OPENCV else '❌ Missing'}<br>
                • Scikit-learn: {'✅ Available' if HAS_SKLEARN else '❌ Missing'}<br>
                • Enhanced UI: {'✅ Available' if HAS_OPTION_MENU else '❌ Basic'}
            </div>
        </div>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.3);">
            <strong>Mode:</strong> {'Production Ready' if HAS_TENSORFLOW and HAS_OPENCV else 'Demo Mode'} | 
            <strong>Updated:</strong> {datetime.now().strftime('%Y-%m-%d')}
        </div>
    </div>
    """

# Enhanced error handling for specific functions
def safe_image_processing(image):
    """Safely process image with comprehensive error handling"""
    try:
        # Validate image first
        is_valid, message = validate_image(image)
        if not is_valid:
            return None, None, message
        
        # Process image
        processed_image, img_resized = preprocess_image(image)
        
        if processed_image is None:
            return None, None, "Failed to preprocess image"
        
        return processed_image, img_resized, "Success"
    
    except Exception as e:
        error_msg = f"Image processing error: {str(e)}"
        st.error(error_msg)
        return None, None, error_msg

def safe_prediction(model, image):
    """Safely make predictions with error handling"""
    try:
        predictions, uncertainties = predict_with_uncertainty(model, image)
        
        # Validate predictions
        if predictions is None or uncertainties is None:
            raise ValueError("Prediction returned None values")
        
        if len(predictions) != 14 or len(uncertainties) != 14:
            raise ValueError(f"Expected 14 predictions, got {len(predictions)}")
        
        # Ensure predictions are in valid range
        predictions = np.clip(predictions, 0, 1)
        uncertainties = np.clip(uncertainties, 0, 1)
        
        return predictions, uncertainties, "Success"
    
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        st.error(error_msg)
        # Return fallback predictions
        predictions, uncertainties = create_sample_predictions()
        return predictions, uncertainties, error_msg

# Main execution
if __name__ == "__main__":
    # Add a header with system status
    if not (HAS_TENSORFLOW and HAS_OPENCV):
        st.info("""
        🔧 **Demo Mode Active**: Some dependencies are not available in this environment. 
        The application will use simulated data for demonstration purposes.
        """)
    
    # Run the main application with error handling
    safe_main()
    
    # Footer
    st.markdown("---")
    st.markdown(
        create_system_info_card(), 
        unsafe_allow_html=True
    )
    
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; margin-top: 2rem;">
        <p>🫁 <strong>Chest X-ray AI Diagnostics</strong> - Educational Demonstration System</p>
        <p><small>For educational and research purposes only. Not for clinical use.</small></p>
    </div>
    """, unsafe_allow_html=True)