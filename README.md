# Realistic-Image-Generation-from-Sketches-using-GAN
Overview
This project generates realistic human face images from hand-drawn sketches using Generative Adversarial Networks (GANs). The model learns the mapping between sketch images and real images to convert grayscale sketches into photorealistic RGB images.

A U-Net based generator preserves the structure of the sketch while adding textures and details. A PatchGAN discriminator helps improve the realism and quality of generated images.

Features

Converts sketches into realistic face images

Uses GAN architecture for image generation

Preserves facial structure using U-Net

Improves image quality using adversarial training

Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

Deep Learning

Dataset

The model is trained on a Face Sketch Dataset containing sketch images and corresponding real images.

Dataset size: ~22,000 images

Image size: 256 × 256

Project Workflow

Data preprocessing and image resizing

Training the GAN model with sketch-image pairs

Generator creates realistic images from sketches

Discriminator evaluates generated images

Model improves through adversarial learning

Applications

Digital art and illustration

Animation and character design

Computer vision research

Future Improvements

Train with larger datasets

Improve image resolution

Extend model to other object categories
