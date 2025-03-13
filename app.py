import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas

DEFAULT_IMAGE_PATH = "assets/Joseph_Fourier.jpg"
MEDICAL_IMAGE_PATH = "assets/Brain.png"

# Helper functions to generate predefined masks
def create_spokes_mask(resolution, num_spokes=10):
    mask = np.zeros((resolution, resolution))
    center = resolution // 2
    mask[center, center] = 1  # Always include the center
    for i in range(num_spokes):
        angle = 2 * np.pi * i / num_spokes
        for r in range(resolution):
            x = int(center + r * np.cos(angle))
            y = int(center + r * np.sin(angle))
            if 0 <= x < resolution and 0 <= y < resolution:
                mask[y, x] = 1
    return mask

def create_spiral_mask(resolution, num_turns=5):
    mask = np.zeros((resolution, resolution))
    center = resolution // 2
    mask[center, center] = 1  # Always include the center
    for t in np.linspace(0, 2 * np.pi * num_turns, resolution * num_turns):
        r = int((t / (2 * np.pi * num_turns))**2 * center)
        x = int(center + r * np.cos(t))
        y = int(center + r * np.sin(t))
        if 0 <= x < resolution and 0 <= y < resolution:
            mask[y, x] = 1
    return mask

def create_low_freq_mask(resolution, radius=10):
    mask = np.zeros((resolution, resolution))
    center = resolution // 2
    y, x = np.ogrid[:resolution, :resolution]
    mask_area = (x - center)**2 + (y - center)**2 <= radius**2
    mask[mask_area] = 1
    mask[center, center] = 1  # Always include the center
    return mask

def create_high_freq_mask(resolution, radius=10):
    mask = np.ones((resolution, resolution))
    center = resolution // 2
    y, x = np.ogrid[:resolution, :resolution]
    mask_area = (x - center)**2 + (y - center)**2 <= radius**2
    mask[mask_area] = 0
    mask[center, center] = 1  # Always include the center
    return mask

def create_horizontal_lines_mask(resolution, skip=2):
    mask = np.zeros((resolution, resolution))
    mask[::skip, :] = 1
    center = resolution // 2
    mask[center, center] = 1  # Always include the center
    return mask

def create_vertical_lines_mask(resolution, skip=2):
    mask = np.zeros((resolution, resolution))
    mask[:, ::skip] = 1
    center = resolution // 2
    mask[center, center] = 1  # Always include the center
    return mask

def reset_mask():
    st.session_state.mask = None

# Load and display an image
st.title('2D Fourier Transform Masking App')

image_option = st.radio(
    "Choose image source", 
    (
        "Default Image", 
        "Medical Image", 
        "Upload your own"
    )
)

if image_option == "Default Image":
    img = Image.open(DEFAULT_IMAGE_PATH).convert('L')

elif image_option == "Medical Image":
    img = Image.open(MEDICAL_IMAGE_PATH).convert('L')

else:
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        img = Image.open(uploaded_image).convert('L')
    else:
        st.warning("Please upload an image or select the default option.")
        st.stop()

img_array = np.array(img)



if img_array is not None:
    max_value = 512
    resolution = st.slider(
        "Adjust resolution for easier drawing", 
        min_value=64, 
        max_value=max_value, 
        value=min(img_array.shape[0], max_value), 
        step=64,
        on_change=reset_mask
    )
    img_resized = np.array(img.resize((resolution, resolution), resample=Image.Resampling.NEAREST))

    f_transform = np.fft.fft2(img_resized)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img_resized, caption='Original Image', use_container_width=True)

    with col2:
        fig, ax = plt.subplots()
        ax.imshow(magnitude_spectrum, cmap='gray')
        ax.set_title('Fourier Magnitude Spectrum')
        st.pyplot(fig)

    if 'show_canvas' not in st.session_state:
        st.session_state.show_canvas = False
    if 'mask' not in st.session_state:
        st.session_state.mask = None

    mask_option = st.selectbox(
        "Choose mask type", 
        [
            "Draw your own", 
            "Spokes",
            "Spiral",
            "Low Frequencies", 
            "High Frequencies", 
            "Horizontal Lines", 
            "Vertical Lines"
        ]
    )

    if mask_option == "Draw your own":
        st.session_state.show_canvas = True

        brush_size = st.slider("Brush Size", min_value=1, max_value=40, value=10)

        st.markdown("### Draw a mask")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=brush_size,
            stroke_color="#000000",
            background_color="#FFFFFF",
            width=resolution,
            height=resolution,
            drawing_mode="freedraw",
            key=f"canvas_{resolution}"
        )

        if canvas_result.image_data is not None and st.button("Confirm Drawing"):
            mask = np.mean(canvas_result.image_data[:, :, :3], axis=2) < 128
            st.session_state.mask = mask.astype(float)

    elif mask_option == "Spokes":
        num_spokes = st.slider("Number of Spokes", min_value=1, max_value=100, value=10)
        st.session_state.mask = create_spokes_mask(resolution, num_spokes)

    elif mask_option == "Spiral":
        num_turns = st.slider("Number of Turns", min_value=1, max_value=100, value=10)
        st.session_state.mask = create_spiral_mask(resolution, num_turns)

    elif mask_option == "Low Frequencies":
        radius = st.slider("Low Frequency Radius", min_value=1, max_value=resolution//2, value=10)
        st.session_state.mask = create_low_freq_mask(resolution, radius)

    elif mask_option == "High Frequencies":
        radius = st.slider("High Frequency Cutoff Radius", min_value=1, max_value=resolution//2, value=10)
        st.session_state.mask = create_high_freq_mask(resolution, radius)

    elif mask_option == "Horizontal Lines":
        skip = st.slider("Skip lines", min_value=0, max_value=10, value=1)
        st.session_state.mask = create_horizontal_lines_mask(resolution, skip=skip+1)

    elif mask_option == "Vertical Lines":
        skip = st.slider("Skip lines", min_value=0, max_value=10, value=1)
        st.session_state.mask = create_vertical_lines_mask(resolution, skip=skip+1)

    if st.session_state.mask is not None:
        if st.button("Invert Drawing"):
            st.session_state.mask = 1 - st.session_state.mask

        masked_f_shift = f_shift * st.session_state.mask

        f_ishift = np.fft.ifftshift(masked_f_shift)
        img_reconstructed = np.fft.ifft2(f_ishift)
        img_reconstructed = np.abs(img_reconstructed)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("### Sampling Pattern")
            st.image(st.session_state.mask, clamp=True, use_container_width=True)

        with col4:
            st.markdown("### Reconstructed Image")
            st.image(img_reconstructed / np.max(img_reconstructed), clamp=True, use_container_width=True)

