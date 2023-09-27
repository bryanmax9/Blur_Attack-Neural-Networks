# Neural Network Blur Attack

![Neural Network Blur Attack Banner](https://i.imgur.com/dVHyyZD.png) 
> *In the Image we can see how the machine wasn't able to detect the image due to the the adversarial attack by intensifying the blur and adding some noise to the image pixels*

An adversarial example generator leveraging sophisticated blurring techniques to challenge the robustness of neural networks.

## 🚀 Overview

The project demonstrates the susceptibility of pretrained neural networks, such as ResNet18, to adversarial attacks. Using a blend of intensified Gaussian blurring, brightness manipulation, and noise augmentation, it highlights the divergence between human and machine perception. The goal is to maintain human interpretability of images while confusing the AI model.

## 🛠 Features

- **Blurring Techniques**: Uses advanced Gaussian blurring to distort the image.
- **Noise Augmentation**: Random noise interference to further trick the model.
- **Brightness Manipulation**: Enhances brightness, making it difficult for the model to recognize.
- **Grayscale Conversion**: Alters the image's color space, adding an extra layer of complexity.

## 🔧 Setup & Usage

1. **Dependencies**:
   ```bash
   pip install torch torchvision matplotlib pillow
## 🔧 Setup & Usage

1. **Run**:
   Navigate to the project directory and execute:
   ```bash
   python blurAttack.py
   ```

 
## Output:
    The script will display the original and attacked images side by side, comparing the neural network's predictions.


