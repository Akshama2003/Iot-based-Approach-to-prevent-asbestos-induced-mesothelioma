# Asbestos Detection and Mesothelioma Risk Analysis System
# For Google Colab - Run this entire code block

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import datetime
import cv2
import PIL
from PIL import Image
import warnings
import io
import gc
import json
import requests
import random
from google.colab import drive, output, files
import base64
import time
import IPython.display
from ipywidgets import widgets
from IPython.display import display, clear_output, HTML

# Suppress warnings
warnings.filterwarnings('ignore')

# Mount Google Drive for data storage
drive.mount('/content/drive', force_remount=True)

# Create necessary directories
base_dir = '/content/asbestos_detection'
os.makedirs(base_dir, exist_ok=True)
model_dir = os.path.join(base_dir, 'models')
os.makedirs(model_dir, exist_ok=True)
data_dir = os.path.join(base_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
reports_dir = os.path.join(base_dir, 'reports')
os.makedirs(reports_dir, exist_ok=True)

print("✅ Initial setup complete")

# Synthetic data generation for demonstration purposes
def generate_synthetic_data():
    print("Generating synthetic training data...")
    
    # Create directories for synthetic data
    train_dir = os.path.join(data_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'asbestos'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'non_asbestos'), exist_ok=True)
    
    # Generate synthetic asbestos images (white fibrous patterns on dark background)
    for i in range(100):
        img_size = 224
        # Create a dark background
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        img.fill(50)  # Dark gray background
        
        # Add white fibrous structures (typical of asbestos)
        num_fibers = np.random.randint(5, 15)
        for _ in range(num_fibers):
            start_x, start_y = np.random.randint(0, img_size, 2)
            length = np.random.randint(20, 80)
            thickness = np.random.randint(1, 4)
            angle = np.random.rand() * 360
            
            # Calculate end point
            end_x = int(start_x + length * np.cos(np.radians(angle)))
            end_y = int(start_y + length * np.sin(np.radians(angle)))
            
            # Draw the fiber
            cv2.line(img, (start_x, start_y), (end_x, end_y), (200, 200, 200), thickness)
        
        # Add some noise
        noise = np.random.randint(0, 30, size=img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Save the image
        pil_img = Image.fromarray(img)
        pil_img.save(os.path.join(train_dir, 'asbestos', f'asbestos_{i}.jpg'))
    
    # Generate non-asbestos images (random textures without fibrous patterns)
    for i in range(100):
        img_size = 224
        # Create a textured background
        img = np.random.randint(30, 120, (img_size, img_size, 3), dtype=np.uint8)
        
        # Add some blobs/particles (not fibrous)
        num_particles = np.random.randint(5, 20)
        for _ in range(num_particles):
            center_x, center_y = np.random.randint(0, img_size, 2)
            radius = np.random.randint(3, 10)
            color = tuple(np.random.randint(100, 200, 3).tolist())
            cv2.circle(img, (center_x, center_y), radius, color, -1)
        
        # Add some noise
        noise = np.random.randint(0, 20, size=img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Apply blur to make it look more natural
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Save the image
        pil_img = Image.fromarray(img)
        pil_img.save(os.path.join(train_dir, 'non_asbestos', f'non_asbestos_{i}.jpg'))
    
    print("✅ Synthetic data generation complete.")

# Generate synthetic environmental data for time series analysis
def generate_environmental_data():
    print("Generating synthetic environmental data...")
    
    # Create a dataframe with timestamps and environmental factors
    start_date = datetime.datetime.now() - datetime.timedelta(days=365)
    dates = [start_date + datetime.timedelta(days=i) for i in range(365)]
    
    # Initialize data dictionary
    data = {
        'date': dates,
        'temperature': [],
        'humidity': [],
        'asbestos_concentration': [],
        'exposure_duration': [],
        'ventilation_rate': [],
        'risk_score': []
    }
    
    # Generate data with seasonal patterns and trends
    for i in range(365):
        # Temperature: seasonal pattern (higher in summer)
        temp_base = 22 + 10 * np.sin(2 * np.pi * i / 365)
        temp = temp_base + np.random.normal(0, 2)
        
        # Humidity: inverse correlation with temperature + random noise
        humidity_base = 60 - 0.5 * (temp_base - 22)
        humidity = humidity_base + np.random.normal(0, 5)
        humidity = max(30, min(humidity, 95))
        
        # Asbestos concentration: some correlation with temperature and ventilation
        # Simulating higher concentrations when maintenance work is done (random spikes)
        is_maintenance = random.random() < 0.05  # 5% chance of maintenance day
        base_concentration = 0.1 + 0.05 * np.sin(2 * np.pi * i / 365 + np.pi)  # Seasonal pattern
        spike = 0.4 if is_maintenance else 0
        concentration = base_concentration + spike + np.random.normal(0, 0.03)
        concentration = max(0.01, concentration)
        
        # Ventilation rate: better in newer systems, worse in older
        ventilation = 80 + np.random.normal(0, 5)
        
        # Exposure duration: working hours but with some variation
        exposure = 8 + np.random.normal(0, 0.5)
        
        # Risk score: composite score based on concentration and exposure
        risk = concentration * exposure * (1 - ventilation/100) * 10
        
        # Add to data dictionary
        data['temperature'].append(temp)
        data['humidity'].append(humidity)
        data['asbestos_concentration'].append(concentration)
        data['exposure_duration'].append(exposure)
        data['ventilation_rate'].append(ventilation)
        data['risk_score'].append(risk)
    
    # Create DataFrame
    env_df = pd.DataFrame(data)
    
    # Save to CSV
    env_df.to_csv(os.path.join(data_dir, 'environmental_data.csv'), index=False)
    
    print("✅ Environmental data generation complete.")
    
    return env_df

# Building the asbestos detection model
def build_asbestos_detection_model(img_height=224, img_width=224):
    # Use MobileNetV2 as base model (efficient for edge devices)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

# Function to train the model on synthetic data
def train_model(batch_size=32, epochs=10):
    print("Training asbestos detection model...")
    
    # Image data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    # Build the model
    model = build_asbestos_detection_model()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'asbestos_model_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,  # Reduced for demo
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save the final model
    model.save(os.path.join(model_dir, 'asbestos_detection_model.h5'))
    
    # Generate and save training metrics plots
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'training_metrics.png'))
    display(plt.gcf())
    plt.close()
    
    print("✅ Model training complete.")
    return model

# Build a time series forecasting model for risk prediction
def build_risk_prediction_model(df):
    print("Building risk prediction model...")
    
    # Prepare features and target
    features = ['temperature', 'humidity', 'asbestos_concentration', 'exposure_duration', 'ventilation_rate']
    X = df[features].values
    y = df['risk_score'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Simple model for demo purposes
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(len(features),)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Regression output
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=25,  # Reduced for demo
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]
    )
    
    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {test_mae:.4f}")
    
    # Save the model
    model.save(os.path.join(model_dir, 'risk_prediction_model.h5'))
    
    # Generate prediction vs actual plot
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Risk Score')
    plt.ylabel('Predicted Risk Score')
    plt.title('Risk Prediction Model Performance')
    plt.savefig(os.path.join(reports_dir, 'risk_prediction_performance.png'))
    display(plt.gcf())
    plt.close()
    
    print("✅ Risk prediction model built and saved.")
    return model

# Function to process a single image and predict asbestos presence
def predict_asbestos(img_path, model):
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Make prediction
    prediction = model.predict(img_array)[0][0]
    
    return float(prediction)

# Function to evaluate mesothelioma risk based on asbestos detection and environmental data
def evaluate_mesothelioma_risk(asbestos_probability, env_data):
    # Extract the latest environmental data
    latest_data = env_data.iloc[-1]
    
    # Create feature vector for risk model
    features = np.array([[
        latest_data['temperature'],
        latest_data['humidity'],
        latest_data['asbestos_concentration'] * (0.5 + 0.5 * asbestos_probability),  # Adjust concentration based on detection
        latest_data['exposure_duration'],
        latest_data['ventilation_rate']
    ]])
    
    # Load risk prediction model
    risk_model = tf.keras.models.load_model(os.path.join(model_dir, 'risk_prediction_model.h5'))
    
    # Predict risk
    risk_score = float(risk_model.predict(features)[0][0])
    
    # Define risk levels
    if risk_score < 0.5:
        risk_level = "Low"
    elif risk_score < 1.0:
        risk_level = "Moderate"
    elif risk_score < 2.0:
        risk_level = "High"
    else:
        risk_level = "Critical"
    
    risk_info = {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "asbestos_probability": asbestos_probability,
        "environmental_factors": {
            "temperature": float(latest_data['temperature']),
            "humidity": float(latest_data['humidity']),
            "ventilation_rate": float(latest_data['ventilation_rate']),
            "exposure_duration": float(latest_data['exposure_duration'])
        }
    }
    
    return risk_info

# Create a class to store analysis data
class AnalysisData:
    def _init_(self):
        self.history = []
    
    def add_result(self, result):
        self.history.append(result)
    
    def get_history(self):
        return self.history

# Create a global analysis data object
analysis_data = AnalysisData()

# Function to generate a sample image for demonstration purposes
def generate_sample_image():
    img_size = 224
    
    # Randomly decide if this is an asbestos image or not
    is_asbestos = random.random() < 0.3  # 30% chance of being asbestos
    
    if is_asbestos:
        # Create a dark background
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        img.fill(50)  # Dark gray background
        
        # Add white fibrous structures (typical of asbestos)
        num_fibers = np.random.randint(5, 15)
        for _ in range(num_fibers):
            start_x, start_y = np.random.randint(0, img_size, 2)
            length = np.random.randint(20, 80)
            thickness = np.random.randint(1, 4)
            angle = np.random.rand() * 360
            
            # Calculate end point
            end_x = int(start_x + length * np.cos(np.radians(angle)))
            end_y = int(start_y + length * np.sin(np.radians(angle)))
            
            # Draw the fiber
            cv2.line(img, (start_x, start_y), (end_x, end_y), (200, 200, 200), thickness)
        
        # Add some noise
        noise = np.random.randint(0, 30, size=img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
    else:
        # Create a textured background
        img = np.random.randint(30, 120, (img_size, img_size, 3), dtype=np.uint8)
        
        # Add some blobs/particles (not fibrous)
        num_particles = np.random.randint(5, 20)
        for _ in range(num_particles):
            center_x, center_y = np.random.randint(0, img_size, 2)
            radius = np.random.randint(3, 10)
            color = tuple(np.random.randint(100, 200, 3).tolist())
            cv2.circle(img, (center_x, center_y), radius, color, -1)
        
        # Add some noise
        noise = np.random.randint(0, 20, size=img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Apply blur to make it look more natural
        img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Save the image to a temporary file
    temp_path = os.path.join(data_dir, 'temp_sample.jpg')
    cv2.imwrite(temp_path, img)
    
    return temp_path, is_asbestos

# Function to generate a report based on collected analysis data
def generate_report():
    history = analysis_data.get_history()
    
    if not history:
        print("No analysis data available. Please run real-time analysis first.")
        return None, None
    
    # Create a DataFrame from history
    df = pd.DataFrame(history)
    
    # Calculate statistics
    stats = {
        "num_samples": len(df),
        "avg_risk_score": df['risk_score'].mean(),
        "max_risk_score": df['risk_score'].max(),
        "avg_asbestos_prob": df['asbestos_probability'].mean(),
        "risk_level_counts": df['risk_level'].value_counts().to_dict(),
    }
    
    # Generate report plots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Risk score over time
    axs[0, 0].plot(range(len(df)), df['risk_score'], marker='o', linestyle='-', color='red')
    axs[0, 0].set_title('Risk Score Trend')
    axs[0, 0].set_xlabel('Sample Index')
    axs[0, 0].set_ylabel('Risk Score')
    axs[0, 0].grid(True)
    
    # Asbestos probability over time
    axs[0, 1].plot(range(len(df)), df['asbestos_probability'], marker='o', color='orange', linestyle='-')
    axs[0, 1].set_title('Asbestos Probability Trend')
    axs[0, 1].set_xlabel('Sample Index')
    axs[0, 1].set_ylabel('Probability')
    axs[0, 1].grid(True)
    
    # Risk level distribution
    risk_levels = ['Low', 'Moderate', 'High', 'Critical']
    risk_counts = [df[df['risk_level'] == level].shape[0] for level in risk_levels]
    colors = ['green', 'yellow', 'orange', 'red']
    axs[1, 0].bar(risk_levels, risk_counts, color=colors)
    axs[1, 0].set_title('Risk Level Distribution')
    axs[1, 0].set_xlabel('Risk Level')
    axs[1, 0].set_ylabel('Count')
    
    # Scatter plot of risk score vs asbestos probability
    axs[1, 1].scatter(df['asbestos_probability'], df['risk_score'], c=df['risk_score'], cmap='viridis', alpha=0.7)
    axs[1, 1].set_title('Risk Score vs Asbestos Probability')
    axs[1, 1].set_xlabel('Asbestos Probability')
    axs[1, 1].set_ylabel('Risk Score')
    axs[1, 1].grid(True)
    
    # Layout adjustments
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(reports_dir, f"asbestos_analysis_report_{timestamp}.png")
    plt.savefig(plot_file)
    
    # Generate HTML report
    html_report = f"""
    <html>
    <head>
        <title>Asbestos Monitoring and Mesothelioma Risk Assessment Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; }}
            .stats {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
            .risk-high {{ color: #e74c3c; font-weight: bold; }}
            .risk-moderate {{ color: #f39c12; font-weight: bold; }}
            .risk-low {{ color: #27ae60; font-weight: bold; }}
            .chart {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Asbestos Monitoring and Mesothelioma Risk Assessment Report</h1>
        <p>Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Analysis Summary</h2>
        <div class="stats">
            <p><strong>Total Samples Analyzed:</strong> {stats['num_samples']}</p>
            <p><strong>Average Risk Score:</strong> {stats['avg_risk_score']:.2f}</p>
            <p><strong>Maximum Risk Score:</strong> {stats['max_risk_score']:.2f}</p>
            <p><strong>Average Asbestos Probability:</strong> {stats['avg_asbestos_prob']:.2f}</p>
        </div>
        
        <h2>Risk Level Distribution</h2>
        <div class="stats">
    """
    
    # Add risk level counts
    for level, count in stats['risk_level_counts'].items():
        css_class = "risk-low"
        if level == "High" or level == "Critical":
            css_class = "risk-high"
        elif level == "Moderate":
            css_class = "risk-moderate"
        
        percentage = (count / stats['num_samples']) * 100
        html_report += f'<p><span class="{css_class}">{level}:</span> {count} samples ({percentage:.1f}%)</p>'
    
    html_report += """
        </div>
        
        <h2>Recommendations</h2>
        <div class="stats">
    """
    
    # Add recommendations based on risk levels
    if stats.get('avg_risk_score', 0) > 1.5:
        html_report += """
            <p class="risk-high">⚠ HIGH RISK DETECTED: Immediate action required</p>
            <ul>
                <li>Implement emergency ventilation procedures</li>
                <li>Provide appropriate PPE to all workers</li>
                <li>Consider temporary suspension of operations in affected areas</li>
                <li>Schedule professional asbestos assessment and removal</li>
                <li>Conduct health screenings for potentially exposed workers</li>
            </ul>
        """
    elif stats.get('avg_risk_score', 0) > 0.8:
        html_report += """
            <p class="risk-moderate">⚠ MODERATE RISK DETECTED: Prompt action recommended</p>
            <ul>
                <li>Increase ventilation in affected areas</li>
                <li>Provide appropriate PPE to workers in high-exposure zones</li>
                <li>Schedule professional assessment within 30 days</li>
                <li>Implement more frequent monitoring</li>
            </ul>
        """
    else:
        html_report += """
            <p class="risk-low">✅ LOW RISK DETECTED: Continue monitoring</p>
            <ul>
                <li>Maintain current safety protocols</li>
                <li>Continue regular monitoring</li>
                <li>Ensure all ventilation systems are properly maintained</li>
                <li>Conduct staff training on asbestos awareness</li>
            </ul>
        """
    
    html_report += """
        </div>
        
        <h2>Detailed Analysis Charts</h2>
        <p>Please refer to the attached images for detailed analysis charts.</p>
    </body>
    </html>
    """
    
    # Save HTML report
    html_report_file = os.path.join(reports_dir, f"asbestos_analysis_report_{timestamp}.html")
    with open(html_report_file, "w") as f:
        f.write(html_report)
    
    print(f"✅ Report generated and saved to {html_report_file}")
    return html_report_file, plot_file

# Data preparation and model training
print("🔄 Setting up the Asbestos Detection and Mesothelioma Risk Analysis System...")

# Generate training data if it doesn't exist
if not os.path.exists(os.path.join(data_dir, 'train')):
    generate_synthetic_data()

# Ensure environmental data exists
env_data_path = os.path.join(data_dir, 'environmental_data.csv')
if os.path.exists(env_data_path):
    env_data = pd.read_csv(env_data_path)
    print("✅ Loaded existing environmental data.")
else:
    env_data = generate_environmental_data()

# Load or train asbestos detection model
try:
    asbestos_model = tf.keras.models.load_model(os.path.join(model_dir, 'asbestos_detection_model.h5'))
    print("✅ Loaded existing asbestos detection model.")
except:
    print("🔄 No existing model found. Training new model...")
    asbestos_model = train_model(epochs=3)  # Reduced epochs for demo

# Load or build risk prediction model
try:
    risk_model = tf.keras.models.load_model(os.path.join(model_dir, 'risk_prediction_model.h5'))
    print("✅ Loaded existing risk prediction model.")
except:
    print("🔄 No existing risk model found. Building new risk model...")
    risk_model = build_risk_prediction_model(env_data)

print("✅ Setup complete! System is ready for analysis.")

# Display title and description
display(HTML("""
<div style="background-color:#4a76cf; padding:20px; border-radius:10px; margin-bottom:20px">
    <h1 style="color:white; text-align:center">Asbestos Detection & Mesothelioma Risk Analysis System</h1>
    <h3 style="color:white; text-align:center">Sugar Mill Industry Specialized Application</h3>
</div>
"""))

# Create a stopping flag for the analysis thread
stop_analysis = False

# Function to run real-time analysis in a thread
def run_real_time_analysis(progress_callback, image_callback, results_callback, iterations=20):
    global stop_analysis
    stop_analysis = False
    
    for i in range(iterations):
        if stop_analysis:
            print("Analysis stopped.")
            break
        
        # Update progress
        progress = (i + 1) / iterations * 100
        progress_callback(progress)
        
        # Generate a sample image
        img_path, is_asbestos = generate_sample_image()
        
        # Display the image
        img = plt.imread(img_path)
        image_callback(img, "Analyzing image...")
        
        # Predict asbestos probability
        asbestos_prob = predict_asbestos(img_path, asbestos_model)
        
        # Evaluate risk
        risk_info = evaluate_mesothelioma_risk(asbestos_prob, env_data)
        
        # Add timestamp
        risk_info["timestamp"] = datetime.datetime.now
