import streamlit as st
from PIL import Image
import torch
import numpy as np
from torchvision import models, transforms
import cv2
from utils import decode_merged_segmap, MERGED_CLASS_NAMES, MERGED_COLORS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
@st.cache_resource
def load_model():
    model = models.segmentation.deeplabv3_resnet50(pretrained=False, aux_loss=True)
    model.classifier[4] = torch.nn.Conv2d(256, 28, kernel_size=1)
    state_dict = torch.load("deeplabv3.pth", map_location=DEVICE)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if not k.startswith("aux_classifier"):
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval().to(DEVICE)
    return model

# Prediction function
def predict(model, img):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)['out']
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    return decode_merged_segmap(pred)

# Page config
st.set_page_config(page_title="Road Scene & Pothole Segmentation", layout="wide")
st.title("🛣 Road Scene Segmentation")

# Sidebar options
mode = st.sidebar.radio("Choose Mode", ["Image Segmentation", "Real-Time Webcam Segmentation"])

with st.sidebar:
    st.markdown("---")
    st.subheader("Legend")
    for i, name in enumerate(MERGED_CLASS_NAMES):
        st.markdown(
            f"<div style='display:flex;align-items:center;'>"
            f"<div style='background-color:rgb{MERGED_COLORS[i]};width:20px;height:20px;"
            f"border-radius:3px;margin-right:10px;'></div>{i}: {name}</div>",
            unsafe_allow_html=True
        )

model = load_model()

# Mode 1: Image Segmentation
if mode == "Image Segmentation":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Original Image", use_container_width=True)

        with st.spinner("Segmenting..."):
            result = predict(model, img)

        st.image(result, caption="Segmented Output (28 Classes)", use_container_width=True)

# Mode 2: Real-Time Webcam Segmentation
elif mode == "Real-Time Webcam Segmentation":
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    st.info("Webcam started. Press Stop to end.")

    stop_btn = st.button("Stop")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_btn:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (512, 512))
        pil_img = Image.fromarray(img_resized)
        result = predict(model, pil_img)

        combined = np.hstack((img_resized, result))
        stframe.image(combined, channels="RGB", use_container_width=True)

    cap.release()
    cv2.destroyAllWindows()