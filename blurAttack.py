import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Load pretrained ResNet18
model = models.resnet18(pretrained=True).eval()

# Load ImageNet classes
with open("imagenet_classes.txt", encoding='utf-8') as f:
    classes = [line.strip() for line in f.readlines()]

# Define transform: resize, convert to tensor, and normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
input_image = Image.open("IrishTerrier.webp")
input_tensor = transform(input_image).unsqueeze(0)

# Predict using the original image
original_output = model(input_tensor)
original_prediction = torch.argmax(original_output, dim=1).item()

# Adversarial Attack: Intensify blur, add noise, adjust brightness and convert to grayscale
blurred_image = input_image.filter(ImageFilter.GaussianBlur(radius=15))
enhancer = ImageEnhance.Brightness(blurred_image)
blurred_image = enhancer.enhance(2.5)  # Drastically increase brightness
blurred_image = ImageOps.grayscale(blurred_image)  # Convert to grayscale
# Convert grayscale back to RGB
blurred_image = ImageOps.colorize(blurred_image, 'black', 'white')
noise = np.random.normal(
    0, 0.2, blurred_image.size[::-1] + (3,)).astype(np.float32)
blurred_image = Image.fromarray(
    (np.array(blurred_image) + (noise * 255)).clip(0, 255).astype(np.uint8))

# Predict using the adversarially attacked image
blurred_tensor = transform(blurred_image).unsqueeze(0)
blurred_output = model(blurred_tensor)
blurred_prediction = torch.argmax(blurred_output, dim=1).item()

# Display both images
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(input_image)
axs[0].set_title(
    f"Original Image - Prediction: {classes[original_prediction]}")
axs[0].axis('off')
axs[1].imshow(blurred_image)
axs[1].set_title(f"Attacked Image - Prediction: {classes[blurred_prediction]}")
axs[1].axis('off')
plt.tight_layout()
plt.show()
