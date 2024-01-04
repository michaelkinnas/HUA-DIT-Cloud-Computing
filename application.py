import torch
import torchvision.transforms as transforms
from timeit import default_timer as timer
from PIL import Image
import io
# from utilities import *

def classify_CIFAR10(image):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    net.eval()

    image = Image.open(io.BytesIO(image)).convert('RGB').resize((32, 32))
    # image = Image.load(image).convert('RGB').resize((32, 32))

    convert_tensor = transforms.ToTensor()
    image = convert_tensor(image).unsqueeze(dim=0)

    with torch.inference_mode():
        preds = net(image)

    probs = torch.sigmoid(preds.squeeze())

    classes_dict = {}
    for key, value in zip(classes, probs):
        classes_dict[key] = value.item()

    return classes_dict