import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Onsight Lite", layout="centered")

st.title("🧗 Onsight Lite")
st.markdown("Upload an artificial climbing wall image and click on a hold to detect its region based on color.")

uploaded_file = st.file_uploader("Upload climbing wall image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.subheader("Click on a hold (just one point)")
    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.3)",
        stroke_width=5,
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="point",
        key="canvas",
    )

    if canvas_result.json_data and canvas_result.json_data["objects"]:
        point = canvas_result.json_data["objects"][-1]
        x, y = int(point["left"]), int(point["top"])
        st.write(f"Clicked at: **({x}, {y})**")

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        target_color = hsv[y, x]

        lower = np.clip(target_color - np.array([10, 50, 50]), 0, 255)
        upper = np.clip(target_color + np.array([10, 50, 50]), 0, 255)

        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

        st.subheader("Detected Region")
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Hold region detected based on color")
    else:
        st.info("Click on a point in the image above to detect the hold.")
else:
    st.warning("Please upload a climbing wall image to begin.")
