import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# -----------------------
# Mini U-Net
# -----------------------

class MiniUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.ReLU(),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128,64,2,stride=2),
            nn.ReLU(),

            nn.Conv2d(64,2,1)
        )

    def forward(self,x):

        x=self.encoder(x)
        x=self.bottleneck(x)
        x=self.decoder(x)

        return x


# -----------------------
# Load Model
# -----------------------

model = MiniUNet()

model.load_state_dict(torch.load("saved_model.pth", map_location="cpu"))

model.eval()

# -----------------------
# UI
# -----------------------

st.title("Automatic Image Colorization")

uploaded = st.file_uploader(
    "Upload a Black & White Image",
    type=["jpg","jpeg","png"]
)

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    image = np.array(image)

    image = cv2.resize(image,(256,256))

    st.subheader("Original")

    st.image(image,use_container_width=True)

    lab = cv2.cvtColor(image,cv2.COLOR_RGB2LAB)

    L = lab[:,:,0]

    L_input = L.astype(np.float32)/255.0

    L_input = np.expand_dims(L_input,0)

    L_input = np.expand_dims(L_input,0)

    tensor = torch.tensor(
        L_input,
        dtype=torch.float32
    )

    with torch.no_grad():

        pred = model(tensor)

    pred = pred.squeeze().numpy()

    A = pred[0]

    B = pred[1]

    A = (A*128)+128
    B = (B*128)+128

    result = np.zeros((256,256,3),dtype=np.uint8)

    result[:,:,0]=L
    result[:,:,1]=A.clip(0,255)
    result[:,:,2]=B.clip(0,255)

    rgb = cv2.cvtColor(
        result,
        cv2.COLOR_LAB2RGB
    )

    st.subheader("Colorized Image")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Original")
        st.image(image)

    with col2:
        st.write("Colorized")
        st.image(rgb)