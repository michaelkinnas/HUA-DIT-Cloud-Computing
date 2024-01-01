import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def classify_CIFAR10(model, image_path):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    image = Image.open(image_path).convert('RGB').resize((32, 32))

    convert_tensor = transforms.ToTensor()
    image = convert_tensor(image).unsqueeze(dim=0)

    model.eval()
    with torch.inference_mode():
        preds = model(image)

    plt.bar(classes, torch.sigmoid(preds.squeeze()))
    plt.show()
