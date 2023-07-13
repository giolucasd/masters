# Semantic Segmentation of Volumetric Medical Images with 3D Convolutional Neural Networks

## 1. Authors

* Alejandra M´arquez Herrera (alejandra.marquez@ucsp.edu.pe)
* Alex Jesus Cuadros-Vargas (alex@ucsp.pe)
* Helio Pedrini (helio@ic.unicamp.br)

## 2. Abstract

A neural network is a mathematical model that is able to perform a task automatically or semi-automatically after learning the human knowledge that we provided.
Moreover, a Convolutional Neural Network (CNN) is a type of neural network that has shown to efficiently learn tasks related to the area of image analysis, such as image segmentation, whose main purpose is to find regions or separable objects within an image.
A more specific type of segmentation, called semantic segmentation, guarantees that each region has a semantic meaning by giving it a label or class.
Since CNNs can automate the task of image semantic segmentation, they have been very useful for the medical area, applying them to the segmentation of organs or abnormalities (tumors).
This work aims to improve the task of binary semantic segmentation of volumetric medical images acquired by Magnetic Resonance Imaging (MRI) using a pre-existing Three-Dimensional Convolutional Neural Network (3D CNN) architecture.
We propose a formulation of a loss function for training this 3D CNN, for improving pixel-wise segmentation results.
This loss function is formulated based on the idea of adapting a similarity coefficient, used for measuring the spatial overlap between the prediction and ground truth, and then using it to train the network.
As contribution, the developed approach achieved good performance in a context where the pixel classes are imbalanced.
We show how the choice of the loss function for training can affect the final quality of the segmentation.
We validate our proposal over two medical image semantic segmentation datasets and show comparisons in performance between the proposed loss function and other pre-existing loss functions used for binary semantic segmentation.

## 3. Notes

### 3.1. I/O

Input: 3D MRI image
Output: Pixel classification over 3D image

### 3.2. Data preprocessing

Resizing, normalization and data augmentation.

### 3.3. Loss function

Common losses for classification are accuracy, precision, recall and F1. Neither have good performance on imbalanced data. MMC and AUC can put up a good fight, but MMC seems convenient since it can measure spacial overlap on images.

### 3.4 Experiments

* NN: Dense V-Net (beat up V-Net, VoxResNet and a MALF-based method)
* Datasets:
    * PROMISE12 Prostate Dataset (50 samples)
    * BRATS15 Brain Tumor Dataset (271 samples)
* Compared losses:
    * cross-entropy loss
    * Dice loss
    * Generalized Wasserstein Dice loss
* Validation:
    * Dice score
    * 70% training, 20% validation, and 10% inference
    * Train from scrath
* Results:
    * MMC seems very solid in comparison with other losses, reaching good performances early and mantaining overall best results.

### 3.4 Questions

* Wouldn't it be more appropriate to evaluate using cross-valdation given such a small dataset size?
* Would train the models on both datasets (obviously resizing and normalizing data) help or harm the models learning? The problems seems quite correlated to me and I'm not sure specializing is necessarily better.

## 4. Conclusions

* More data and better fitted model for the specific problems could help good performance and better evaluation.
* It might be a good idea to derive loss function from a correlation coefficient.
* Better analysis could be done for bigger batch sizes, but computational power didn't allow it.
* Proposed loss is actually good.