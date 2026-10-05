"""Web demo:  streamlit run app.py   (needs model.joblib from train.py)"""
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from predict import load, predict_array

st.set_page_config(page_title="Bird Species Identifier", page_icon="🐦")
st.title("🐦 Bird Species Identifier")
st.caption("Classical ML + ANN (MLP) on hand-crafted features")
st.info(
    "This model can choose only among the 200 species in CUB-200-2011. "
    "It does not include owls, peacocks, or bats; unsupported animals may be "
    "assigned a misleading nearest species."
)


@st.cache_resource
def get_model():
    return load("model.joblib")


up = st.file_uploader("Upload a bird photo", type=["jpg", "jpeg", "png"])
if up:
    pil = Image.open(up).convert("RGB")
    st.image(pil, use_container_width=True)
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    with st.spinner("Analysing..."):
        top = predict_array(get_model(), bgr, k=5)
    st.subheader(f"Prediction: {top[0][0]}")
    for name, p in top:
        st.write(f"{name} - {p * 100:.1f}%")
        st.progress(min(1.0, p))
