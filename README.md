# Automatic Image Colorization using Mini U-Net

## Overview

This project implements a simple **Deep Learning Image Colorization** model using a **Mini U-Net** architecture in **PyTorch**.

The model learns to predict the **A** and **B** color channels from the **L (Lightness)** channel of an image in the **LAB color space**.

The project follows the encoder-decoder approach discussed in the paper *"Let There Be Color: Deep Learning Image Colorization"* and is implemented as a lightweight educational project.

---

## Features

- RGB → LAB color conversion
- Mini U-Net architecture
- GPU training (CUDA supported)
- Automatic dataset loading
- Model checkpoint saving
- Colorize grayscale images
- Streamlit Web Application

---

## Requirements

- Python 3.10+
- PyTorch
- OpenCV
- NumPy
- Streamlit

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Model Architecture

```
Input (L Channel)

        │

        ▼

Encoder

Conv2D
ReLU

Conv2D
ReLU

MaxPool

        │

        ▼

Bottleneck

Conv2D
ReLU

        │

        ▼

Decoder

ConvTranspose2D

ReLU

Conv2D

        │

        ▼

Predicted A + B Channels
```

---

## Training

graytocolor.ipynb 

I have trained in kaggle

---

## Streamlit Application

Launch the web app:

```bash
streamlit run app.py
```

The application allows users to

- Upload a grayscale image
- View the original image
- Generate a colorized image
- Download the result

---

## Colorization Pipeline

```
RGB Image

        │

        ▼

LAB Conversion

        │

        ▼

L Channel

        │

        ▼

Mini U-Net

        │

        ▼

Predicted A + B

        │

        ▼

LAB Image

        │

        ▼

RGB Image
```

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Streamlit

---

## Future Improvements

- Full U-Net with Skip Connections
- Attention U-Net
- GAN-based Image Colorization
- Higher Resolution Images
- Perceptual Loss
- SSIM and PSNR Evaluation
- Historical Photo Restoration

---

## Results

The model is capable of learning basic semantic color information such as:

- Green trees
- Blue sky
- Gray roads
- Brown soil

The generated colors depend on the quality and size of the training dataset.

---

## References

Justin Olah, Jenny Yang

**Let There Be Color: Deep Learning Image Colorization**

Stanford University

```
This project is developed for educational purposes to demonstrate deep learning-based image colorization using a lightweight Mini U-Net architecture.
```