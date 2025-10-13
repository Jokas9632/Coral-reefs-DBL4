import torch.nn as nn
import torch.nn.functional as F

class CoralCNN(nn.Module):
    """
    Custom CNN model for coral health classification.

    Classes:
        0: Healthy
        1: Bleached/Unhealthy
    """

    def __init__(self, num_classes=2):
        super(CoralCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # assuming input images are 224x224
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)    # output 2 classes

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional blocks with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 224 -> 112
        x = self.pool(F.relu(self.conv2(x)))  # 112 -> 56
        x = self.pool(F.relu(self.conv3(x)))  # 56 -> 28
        x = self.pool(F.relu(self.conv4(x)))  # 28 -> 14

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # raw logits for 2 classes
        return x

# Example usage
model = CoralCNN(num_classes=2)
print(model)
