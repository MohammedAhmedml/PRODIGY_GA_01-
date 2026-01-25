# Task-04: Image-to-Image Translation with cGAN (Pix2Pix) — Maps Dataset

## Objective
Implement an image-to-image translation model using a conditional GAN (cGAN) called Pix2Pix.

This project learns a mapping from Satellite images → Map images using paired training data.

## Dataset
- Maps (pix2pix dataset): paired images provided as a single combined image (A|B) concatenated side-by-side.
- Folders: data/maps/train, data/maps/val, data/maps/test

## Model Architecture
### Generator (G)
- U-Net (encoder–decoder with skip connections)
- Output: fakeB = G(A)

### Discriminator (D)
- PatchGAN (patch-level real/fake classification)
- Real pair: (A, B), Fake pair: (A, G(A))

## Loss Function
Total loss:
L = L_GAN + λ * L1, with λ = 100

## Training Configuration (Colab-friendly)
- Framework: PyTorch
- Image size: 256×256
- Batch size: 4
- Epochs: 10
- Optimizer: Adam (lr=2e-4, betas=(0.5, 0.999))

## Results
Saved in results/ as grids:
Input (A) | Generated (G(A)) | Ground Truth (B)

## How to Run (Google Colab)
Install:
```bash
pip install -r requirements.txt
