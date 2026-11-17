import os
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import models, transforms

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mri_C_both_half.pth")

# Same preprocessing as in training (no augmentation)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_model():
    state = torch.load(MODEL_FILE, map_location="cpu")
    # The file is stored in 16-bit; convert back to normal precision
    state = {k: (v.float() if v.is_floating_point() else v) for k, v in state.items()}
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(state)
    return model.eval()

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠")
st.title("Brain Tumor MRI Classifier (practice project)")
st.warning("Educational demo only. Not a medical device. Do not use it for diagnosis.")
st.caption(
    "ResNet18 fine-tuned on a public brain MRI dataset (glioma, meningioma, no tumor, pituitary). "
    "About 90% on a held-out test split, possibly optimistic because the data has no patient IDs. "
    "Glioma and meningioma are often confused with each other."
)

model = load_model()
file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded image", width=320)
    x = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    scores = {c: float(p) for c, p in zip(CLASSES, probs)}
    top = max(scores, key=scores.get)

    st.subheader(f"Model's top guess: {top} ({scores[top]:.0%} confidence)")
    if scores[top] < 0.60:
        st.error("The model is not confident here. Treat this result as unreliable.")
    st.bar_chart(scores)
    st.info(
        "The model only knows four classes and always picks one of them, even for images "
        "that are not brain MRIs. Confidence is not the same as being correct."
    )
