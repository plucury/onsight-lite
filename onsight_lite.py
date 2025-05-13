import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import io
import colorsys
import base64
from io import BytesIO
from streamlit_image_coordinates import streamlit_image_coordinates

# Configure the page
st.set_page_config(page_title="Onsight Lite (Simple)", layout="centered")

st.title("🧗 Onsight Lite (Simple Version)")
st.markdown("Upload an artificial climbing wall image and use coordinates to detect holds based on color.")

# Function to detect only the specific hold that was clicked
def detect_specific_hold(img_array, click_x, click_y, tolerance=30):
    height, width, _ = img_array.shape
    
    # Get the target color at the clicked point
    target_color = img_array[click_y, click_x]
    r, g, b = target_color
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    
    # Create initial mask based on color similarity
    initial_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Define color similarity threshold
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
                initial_mask[y, x] = 1
    
    # Now perform connected component analysis to only keep the region containing the clicked point
    # First, create a labeled array of the initial mask
    from scipy import ndimage
    labeled_array, num_features = ndimage.label(initial_mask)
    
    # Get the label of the clicked point
    clicked_label = labeled_array[click_y, click_x]
    
    # If the clicked point isn't in a valid region, return empty mask
    if clicked_label == 0:
        return np.zeros((height, width), dtype=np.uint8)
    
    # Create final mask with only the connected component that was clicked
    final_mask = np.zeros((height, width), dtype=np.uint8)
    final_mask[labeled_array == clicked_label] = 255
    
    # Apply smoothing operations to reduce jagged edges
    # 1. Apply a small Gaussian blur to smooth the edges
    final_mask = ndimage.gaussian_filter(final_mask, sigma=0.7)
    
    # 2. Re-threshold to make it binary again
    final_mask = (final_mask > 128).astype(np.uint8) * 255
    
    # 3. Apply morphological operations to further smooth the edges
    # Create a small circular structuring element
    struct = ndimage.generate_binary_structure(2, 2)
    
    # Close small holes (dilate then erode)
    final_mask = ndimage.binary_closing(final_mask, structure=struct, iterations=1).astype(np.uint8) * 255
    
    # Remove small isolated pixels (erode then dilate)
    final_mask = ndimage.binary_opening(final_mask, structure=struct, iterations=1).astype(np.uint8) * 255
    
    return final_mask

# File uploader
uploaded_file = st.file_uploader("Upload climbing wall image", type=["jpg", "jpeg", "png"])

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
            st.write("Uploaded Image:")
            # Convert to base64 for display
            import base64
            from io import BytesIO
            
            # Save image to buffer
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            # Create base64 string
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Initialize session state for coordinates
            if 'x_coord' not in st.session_state:
                st.session_state.x_coord = image.width // 2
            if 'y_coord' not in st.session_state:
                st.session_state.y_coord = image.height // 2
            
            # Display instructions
            st.write("**Click directly on the image to select a point on the climbing hold**")
            
            # Create a copy of the image for drawing
            img_with_marker = image.copy()
            draw = ImageDraw.Draw(img_with_marker)
            
            # Draw a red circle at the current coordinates
            x, y = st.session_state.x_coord, st.session_state.y_coord
            draw.ellipse((x-5, y-5, x+5, y+5), fill="red")
            
            # Display the image with the streamlit-image-coordinates component
            clicked_coords = streamlit_image_coordinates(img_with_marker, key="image_coordinates")
            
            # Update coordinates if image was clicked
            if clicked_coords and clicked_coords != st.session_state.get('last_clicked'):
                st.session_state.last_clicked = clicked_coords
                st.session_state.x_coord = clicked_coords["x"]
                st.session_state.y_coord = clicked_coords["y"]
                st.rerun()
            
            # Display current coordinates
            st.write(f"Current point: (x: {st.session_state.x_coord}, y: {st.session_state.y_coord})")
            
            # Use the selected coordinates for processing
            x, y = st.session_state.x_coord, st.session_state.y_coord
            
            # Button to detect the hold
            if st.button("Detect Hold"):
                # Get the target color at the clicked point
                target_color = img_array[y, x]
                st.write(f"Selected color at ({x}, {y}): RGB{tuple(target_color)}")
                
                # Create a small circle to show the selected point
                point_img = image.copy()
                draw = ImageDraw.Draw(point_img)
                draw.ellipse((x-5, y-5, x+5, y+5), fill="red")
                
                # Detect only the specific hold that was clicked
                mask = detect_specific_hold(img_array, x, y, tolerance=25)
                
                # Create a result array that dims unselected areas instead of making them black
                result_array = img_array.copy().astype(np.float32)
                
                # For unselected areas, reduce brightness and contrast
                dimming_factor = 0.3  # How much to dim unselected areas (0.0 = black, 1.0 = unchanged)
                
                # Create a smoother transition mask
                # First convert binary mask to float for smooth transitions
                smooth_mask = mask.astype(np.float32) / 255.0
                
                # Apply a Gaussian blur to create a soft edge
                from scipy import ndimage
                smooth_mask = ndimage.gaussian_filter(smooth_mask, sigma=2.0)
                
                # Apply the smooth mask for a gradual transition
                for i in range(3):  # Apply to each RGB channel
                    # Create a weighted blend between original and dimmed image
                    dimmed_channel = img_array[:,:,i] * dimming_factor
                    result_array[:,:,i] = smooth_mask * img_array[:,:,i] + (1 - smooth_mask) * dimmed_channel
                
                # Convert back to uint8
                result_array = np.clip(result_array, 0, 255).astype(np.uint8)
                
                # Convert result to PIL Image
                result_img = Image.fromarray(result_array.astype('uint8'))
                
                # Convert to base64 for display
                result_buffered = BytesIO()
                result_img.save(result_buffered, format="PNG")
                result_img_str = base64.b64encode(result_buffered.getvalue()).decode()
                
                # Display the result image
                st.subheader("Detected Region")
                st.markdown(f"""
                <div style="text-align: center;">
                    <img src="data:image/png;base64,{result_img_str}" style="max-width: 100%; height: auto;">
                    <p>Hold region detected based on color</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Save result to a file in memory for download
                img_byte_arr = BytesIO()
                result_img.save(img_byte_arr, format='PNG')
                
                # Display download button for the result
                st.download_button(
                    label="Download Detected Hold Image",
                    data=img_byte_arr.getvalue(),
                    file_name="detected_hold.png",
                    mime="image/png"
                )
    
    except Exception as e:
        st.error(f"An error occurred while processing the image: {str(e)}. Please try another image.")
else:
    st.warning("Please upload a climbing wall image to begin.")
