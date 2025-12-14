# Deep Learning Homework - Airbus Ship Detection

## Team Members
* **[Csáki Márton]** ([R0OQD4])
* **[Marák Levente]** ([K2DE0K])
* **[Ogleznyev Pável]** ([GRKO04])

## Project Overview
This project focuses on the semantic segmentation of ships in satellite images using the **Airbus Ship Detection Challenge** dataset. The goal is to accurately locate ships and create segmentation masks, distinguishing them from the open sea, clouds, and coastline. We implemented a deep learning solution using a custom **U-Net** architecture.

## Key Features

* **Model Architecture:** We designed a custom **U-Net** model from scratch. It features:
    * **Encoder:** 4 blocks of Convolutional layers with Batch Normalization, Max Pooling, and Dropout for robust feature extraction.
    * **Decoder:** Transpose Convolutions for upsampling, concatenated with skip connections from the encoder to preserve spatial details.
    * **Output:** A single-channel Sigmoid activation layer for binary segmentation (ship vs. background).
* **Smart Preprocessing:**
    * **Smart Cropping:** Since the original 768x768 images are sparse (mostly empty sea), we implemented a `get_smart_crop` function that prioritizes cropping regions containing ships during training.
    * **Class Balancing:** We addressed the extreme class imbalance by undersampling empty images, maintaining a specific ratio (e.g., 1:3) between ship-containing and empty images.
* **Data Augmentation:** To improve generalization, the training generator applies random horizontal/vertical flips and 90-degree rotations.
* **Custom Loss Function:** We utilized a combined **BCE + Dice Loss** to handle the pixel imbalance inherent in segmentation tasks.

## Dataset
* **Source:** [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection)
* **Input:** 768x768 RGB satellite images.
* **Labels:** Run-Length Encoded (RLE) masks provided in a CSV file.

## Repository Structure
* `Airbus_Ship_Detection.ipynb`: The main Jupyter Notebook containing the full pipeline:
    * Data loading and RLE decoding.
    * Data generator with smart cropping and augmentation.
    * U-Net model definition (`build_unet_improved`).
    * Training loop with callbacks.
    * Evaluation and visualization.

## Software Environment & Requirements
To reproduce the results, the following environment is recommended (e.g., Google Colab T4/L4 GPU):

* **Python:** 3.x
* **Deep Learning Framework:** TensorFlow / Keras
* **Libraries:**
    * `pandas`, `numpy` (Data manipulation)
    * `opencv-python` (Image processing)
    * `matplotlib` (Visualization)
    * `scikit-learn` (Train/Val split)

##Running the project
Run all cells to train and evaluate the model
