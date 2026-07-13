import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms


class ModelWithAdapter(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.adapter = nn.Conv2d(1, 3, kernel_size=1)
        self.backbone = backbone

    def forward(self, x):
        x = self.adapter(x)
        return self.backbone(x)


def create_model(model_path=None):
    resnet_weights = models.ResNet34_Weights.DEFAULT
    backbone = models.resnet34(weights=resnet_weights)

    backbone.requires_grad_(False)

    backbone.fc = nn.Sequential(
        nn.Linear(512, 128, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(128, 2, bias=True)
    )

    model = ModelWithAdapter(backbone)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Model loaded: {model_path}")

    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def predict_image(model, image_path, device='cpu'):
    image = Image.open(image_path).convert('L')
    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0)

    img_tensor = img_tensor.to(device)
    model = model.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, dim=1)

    class_names = ['NORMAL', 'PNEUMONIA']
    pred_class = class_names[predicted.item()]
    confidence = probabilities[0][predicted].item()

    return {
        'class': pred_class,
        'confidence': confidence,
        'probabilities': {
            class_names[0]: probabilities[0][0].item(),
            class_names[1]: probabilities[0][1].item()
        }
    }


def predict_batch(model, image_paths, device='cpu'):
    results = []
    for path in image_paths:
        result = predict_image(model, path, device)
        result['path'] = path
        results.append(result)
    return results


if __name__ == "__main__":
    MODEL_PATH = "../models/best_model.pth"
    model = create_model(MODEL_PATH)

    result = predict_image(model, "path.jpg")
    print(result)