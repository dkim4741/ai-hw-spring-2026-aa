import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import SimpleMLP

def train():
    # Convert images to tensors
    transform = transforms.ToTensor()

    # Load MNIST training dataset
    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True
    )

    # Create model
    model = SimpleMLP()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {total_loss:.4f}"
        )

    # Save trained model
    torch.save(model.state_dict(), "mnist_mlp.pth")
    print("Training completed.")
    print("Model saved as mnist_mlp.pth")

if __name__ == "__main__":
    train()