import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
import io
import colorsys

# Configure the page
st.set_page_config(page_title="Onsight Lite", layout="centered")

st.title("🧗 Onsight Lite")
st.markdown("Upload an artificial climbing wall image and click on a hold to detect its region based on color.")

# File uploader
uploaded_file = st.file_uploader("Upload climbing wall image", type=["jpg", "jpeg", "png"])

# Function to detect similar colors using PIL instead of OpenCV
def detect_similar_colors(img_array, target_color, tolerance=30):
    height, width, _ = img_array.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Convert target color to HSV for better color matching
    r, g, b = target_color
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    
    for y in range(height):
        for x in range(width):
            pixel = img_array[y, x]
            pr, pg, pb = pixel
            ph, ps, pv = colorsys.rgb_to_hsv(pr/255.0, pg/255.0, pb/255.0)
            
            # Calculate color distance in HSV space with emphasis on hue
            h_diff = min(abs(h - ph), 1 - abs(h - ph)) * 360  # Hue is circular
            s_diff = abs(s - ps)
            v_diff = abs(v - pv)
            
            # Weighted distance
            distance = (h_diff * 0.5) + (s_diff * 0.3 * 100) + (v_diff * 0.2 * 100)
            
            if distance < tolerance:
                mask[y, x] = 255
    
    return mask

# Process the uploaded image
if uploaded_file:
    try:
        # Read the file content
        file_bytes = uploaded_file.getvalue()
        if len(file_bytes) == 0:
            st.error("The uploaded file appears to be empty. Please try another image.")
        else:
            # Open the image with PIL
            image = Image.open(io.BytesIO(file_bytes))
            img_array = np.array(image.convert("RGB"))
            
            # Display the image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Create columns for coordinates input
            col1, col2 = st.columns(2)
            with col1:
                x = st.number_input("X coordinate", min_value=0, max_value=image.width-1, value=image.width//2)
            with col2:
                y = st.number_input("Y coordinate", min_value=0, max_value=image.height-1, value=image.height//2)
            
            # Button to detect the hold
            if st.button("Detect Hold"):
                # Get the target color at the clicked point
                target_color = img_array[y, x]
                st.write(f"Selected color at ({x}, {y}): RGB{tuple(target_color)}")
                
                # Create a small circle to show the selected point
                point_img = image.copy()
                draw = ImageDraw.Draw(point_img)
                draw.ellipse((x-5, y-5, x+5, y+5), fill="red")
                st.image(point_img, caption="Selected Point", use_column_width=True)
                
                # Detect similar colors
                mask = detect_similar_colors(img_array, target_color)
                
                # Apply the mask to the original image
                result_array = np.zeros_like(img_array)
                for i in range(3):  # Apply to each RGB channel
                    result_array[:,:,i] = np.where(mask == 255, img_array[:,:,i], 0)
                
                # Convert result back to PIL Image
                result_img = Image.fromarray(result_array.astype('uint8'))
                
                # Display the result
                st.subheader("Detected Region")
                st.image(result_img, caption="Hold region detected based on color")
    
    except UnidentifiedImageError:
        st.error("Could not identify the image format. Please ensure you're uploading a valid JPG, JPEG, or PNG file.")
    except Exception as e:
        st.error(f"An error occurred while processing the image: {str(e)}. Please try another image.")
else:
    st.warning("Please upload a climbing wall image to begin.")
