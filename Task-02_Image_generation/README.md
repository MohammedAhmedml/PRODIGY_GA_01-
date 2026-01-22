# Task-02: Image Generation with Pre-trained Models

## Objective
Generate images from text prompts using a pre-trained generative model (Stable Diffusion) and document prompts and parameters.

## Model/Tool
- Model: runwayml/stable-diffusion-v1-5
- Library: Hugging Face diffusers
- Platform: Google Colab (GPU)

## Parameters
- num_inference_steps: 30
- guidance_scale (CFG): 7.5
- Negative prompt: blurry, low quality, distorted, watermark, text, extra fingers, extra limbs

## Outputs
Images are stored in `outputs/`.
Documentation is in `prompts_and_settings.txt`.
(Optional) notebook is in `notebook/`.
