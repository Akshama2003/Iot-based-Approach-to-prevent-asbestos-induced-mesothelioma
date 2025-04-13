import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from skimage import measure
from scipy import ndimage
from PIL import Image
import io

# Set up Streamlit page
st.set_page_config(page_title="Asbestos Fiber Analyzer", layout="wide")
st.title("Real-Time Asbestos Fiber Analysis")
st.markdown("""
This tool analyzes microscopy images for asbestos fibers, measuring their dimensions and classifying fiber types.
Upload an image with scale markers (2μm or 100nm) for accurate measurements.
""")


class AsbestosAnalyzer:
    def __init__(self):
        # Initialize the model
        self.model = self.build_model()
        self.pixel_to_um = 0.0  # Will be set based on scale markers

    def build_model(self):
        """Build a simplified UNet-like model for fiber segmentation"""
        inputs = tf.keras.Input(shape=(256, 256, 3))

        # Downsample path
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)

        # Bottleneck
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)

        # Upsample path
        x = layers.UpSampling2D(2)(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.UpSampling2D(2)(x)
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)

        # Output layer
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def calibrate_scale(self, image):
        """Detect scale markers and calculate pixels-to-microns ratio"""
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Find contours (assuming scale markers are the largest horizontal lines)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        horizontal_lines = []

        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            width = max(rect[1])
            angle = rect[2]

            # Filter for horizontal lines (scale markers)
            if width > 50 and abs(angle) < 10:
                horizontal_lines.append((width, cnt))

        if len(horizontal_lines) >= 2:
            # Sort by length and take the two largest (assuming these are scale bars)
            horizontal_lines.sort(reverse=True)
            pixel_length1, cnt1 = horizontal_lines[0]
            pixel_length2, cnt2 = horizontal_lines[1]

            # Get the scale (2μm and 100nm markers)
            if pixel_length1 > pixel_length2:
                self.pixel_to_um = 2.0 / pixel_length1  # 2μm scale
                cv2.drawContours(image, [cnt1], -1, (0, 255, 0), 2)
                cv2.putText(image, "2μm scale detected", (int(cnt1[0][0][0]), int(cnt1[0][0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                self.pixel_to_um = 0.1 / pixel_length2  # 100nm scale
                cv2.drawContours(image, [cnt2], -1, (0, 255, 0), 2)
                cv2.putText(image, "100nm scale detected", (int(cnt2[0][0][0]), int(cnt2[0][0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            st.session_state.scale_calibrated = True
        else:
            st.warning("Could not detect scale markers. Using default scale (0.1μm/pixel).")
            self.pixel_to_um = 0.1
            st.session_state.scale_calibrated = False

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        resized = cv2.resize(image, (256, 256))
        normalized = resized / 255.0
        return np.expand_dims(normalized, axis=0)

    def analyze_fibers(self, image):
        """Main analysis function"""
        if self.pixel_to_um == 0.0:
            self.calibrate_scale(image)

        processed = self.preprocess_image(image)
        mask = self.model.predict(processed, verbose=0)[0]

        _, binary_mask = cv2.threshold((mask * 255).astype(np.uint8), 128, 255, cv2.THRESH_BINARY)
        labeled_image, num_features = ndimage.label(binary_mask)
        properties = measure.regionprops(labeled_image)

        results = []
        for prop in properties:
            if prop.area < 10:
                continue

            length = prop.major_axis_length * self.pixel_to_um
            width = prop.minor_axis_length * self.pixel_to_um
            aspect_ratio = length / max(width, 0.001)

            if aspect_ratio > 10 and width < 0.5:
                fiber_type = "Possible asbestos fiber"
                color = (0, 0, 255)  # Red
            elif aspect_ratio > 5:
                fiber_type = "Suspected fiber"
                color = (0, 165, 255)  # Orange
            else:
                fiber_type = "Non-asbestos particle"
                color = (0, 255, 0)  # Green

            results.append({
                'length_um': length,
                'width_um': width,
                'aspect_ratio': aspect_ratio,
                'type': fiber_type,
                'color': color,
                'centroid': prop.centroid,
                'coords': prop.coords
            })

        return results

    def visualize_results(self, image, results):
        """Draw analysis results on the image"""
        display_img = image.copy()
        display_img = cv2.resize(display_img, (512, 512))
        scale_x = 512 / image.shape[1]
        scale_y = 512 / image.shape[0]

        for fiber in results:
            coords = fiber['coords']
            scaled_coords = np.array([[x * scale_x, y * scale_y] for y, x in coords])
            cv2.fillPoly(display_img, [scaled_coords.astype(int)], fiber['color'])

            y, x = fiber['centroid']
            cv2.circle(display_img, (int(x * scale_x), int(y * scale_y)), 5, (255, 255, 255), -1)

            label = f"{fiber['type']} L:{fiber['length_um']:.1f}μm W:{fiber['width_um']:.1f}μm"
            cv2.putText(display_img, label, (int(x * scale_x) + 10, int(y * scale_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return display_img


@st.cache_resource
def load_analyzer():
    return AsbestosAnalyzer()


analyzer = load_analyzer()

# File uploader
uploaded_file = st.file_uploader("Upload microscopy image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    if image_np.ndim == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    elif image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing fibers..."):
            results = analyzer.analyze_fibers(image_np)
            result_img = analyzer.visualize_results(image_np, results)

            with col2:
                st.subheader("Analysis Results")
                st.image(result_img, use_column_width=True, channels="BGR")

                st.subheader("Fiber Statistics")
                if results:
                    fiber_counts = {}
                    for fiber in results:
                        fiber_counts[fiber['type']] = fiber_counts.get(fiber['type'], 0) + 1

                    st.write("**Fiber/particle counts:**")
                    for fiber_type, count in fiber_counts.items():
                        st.write(f"- {fiber_type}: {count}")

                    avg_length = np.mean([f['length_um'] for f in results])
                    avg_width = np.mean([f['width_um'] for f in results])
                    st.write(f"**Average length:** {avg_length:.2f} μm")
                    st.write(f"**Average width:** {avg_width:.2f} μm")

                    with st.expander("Show detailed fiber measurements"):
                        for i, fiber in enumerate(results, 1):
                            st.write(f"**Fiber {i}:** {fiber['type']}")
                            st.write(f"- Length: {fiber['length_um']:.2f} μm")
                            st.write(f"- Width: {fiber['width_um']:.2f} μm")
                            st.write(f"- Aspect ratio: {fiber['aspect_ratio']:.1f}")
                else:
                    st.success("No fibers or particles detected above threshold.")

# Create sample image in memory
sample_img = np.zeros((400, 600, 3), dtype=np.uint8)
cv2.rectangle(sample_img, (50, 350), (150, 360), (255, 255, 255), -1)
cv2.putText(sample_img, "2μm", (80, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.rectangle(sample_img, (50, 380), (60, 390), (255, 255, 255), -1)
cv2.putText(sample_img, "100nm", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.line(sample_img, (200, 100), (400, 120), (200, 200, 200), 2)
cv2.line(sample_img, (300, 200), (450, 180), (200, 200, 200), 1)

# Convert to bytes for download
_, buffer = cv2.imencode(".png", sample_img)
io_buf = io.BytesIO(buffer)

# Sidebar
st.sidebar.title("About")
st.sidebar.info("""
This tool analyzes microscopy images for asbestos fibers using computer vision and machine learning.

**How to use:**
1. Upload an image with visible scale markers (2μm or 100nm)
2. Click "Analyze Image" button
3. View detected fibers and measurements
""")

st.sidebar.download_button(
    label="Download sample image",
    data=io_buf,
    file_name="sample_asbestos_image.png",
    mime="image/png"
)
