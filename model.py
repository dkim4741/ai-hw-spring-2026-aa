import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Define MLP architecture
        self.mlp_model = nn.Sequential(
            # Convert 28x28 image to 784 values
            nn.Flatten(),

            # Hidden layer (784 -> 128)
            nn.Linear(28 * 28, 128),

            # Activation function
            nn.ReLU(),

            # Output layer (10 digit classes)
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # Pass input through the network
        return self.mlp_model(x)