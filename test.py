import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import SimpleMLP


def fgsm_attack(image, attack_strength, image_gradient):
    image = image + attack_strength * image_gradient.sign()
    return torch.clamp(image, 0, 1)


def test():
    # Convert images to tensors
    transform = transforms.ToTensor()

    # Load data and model weights directly
    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False
    )

    # Create model
    model = SimpleMLP()

    # Load trained weights
    model.load_state_dict(torch.load("mnist_mlp.pth"))
    model.eval()

    normal_correct = 0
    attacked_correct = 0
    total = 0

    attack_strength = 0.15
    criterion = nn.CrossEntropyLoss()

    for images, labels in test_loader:
        images.requires_grad = True

        # Original Accuracy
        outputs = model(images)
        normal_correct += (outputs.argmax(dim=1) == labels).sum().item()

        # FGSM Attack
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        attacked_images = fgsm_attack(images, attack_strength, images.grad.data)

        attacked_predictions = model(attacked_images)
        attacked_correct += (attacked_predictions.argmax(dim=1) == labels).sum().item()

        total += labels.size(0)

    print(f"[Result] Accuracy: {100 * normal_correct / total:.2f}%")
    print(f"[FGSM] Accuracy: {100 * attacked_correct / total:.2f}%")
    print(f"[ASR] Attack Success Rate: {100 * (normal_correct - attacked_correct) / normal_correct:.2f}%")

if __name__ == "__main__":
    test()