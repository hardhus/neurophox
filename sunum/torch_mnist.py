import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from neurophox.simulation import RMNumpyLayer

# Hyperparameters
epochs = 200
batch_size = 512
learning_rate = 0.001
N_classes = 10

# Data preprocessing (Fourier transform function)
def fourier_transform(data, scale):
    data = data.numpy()
    fft_data = np.fft.fft2(data)
    fft_data = np.fft.fftshift(fft_data)
    scaled_fft_data = np.abs(fft_data) ** scale
    return torch.tensor(scaled_fft_data, dtype=torch.float32)

# Load and preprocess MNIST data
transform = transforms.Compose([
    transforms.ToTensor()
])
mnist_train = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# Apply Fourier transform to the data
scale = 3
train_data = torch.stack([fourier_transform(img[0][0], scale) for img in mnist_train])
train_labels = torch.tensor(mnist_train.targets)

test_data = torch.stack([fourier_transform(img[0][0], scale) for img in mnist_test])
test_labels = torch.tensor(mnist_test.targets)

train_loader = DataLoader(dataset=list(zip(train_data, train_labels)), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=list(zip(test_data, test_labels)), batch_size=batch_size, shuffle=False)

# Model definition
class ONNWithRMNumpy(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ONNWithRMNumpy, self).__init__()
        self.rm_layer1 = RMNumpyLayer(input_dim, 128, nonlinearity="relu")
        self.rm_layer2 = RMNumpyLayer(128, 64, nonlinearity="relu")
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.rm_layer1(x)
        x = self.rm_layer2(x)
        x = self.fc(x)
        return x

input_dim = train_data[0].numel()  # Flattened input dimension
model = ONNWithRMNumpy(input_dim, N_classes).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train_model():
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Evaluation loop
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

# Run the training and evaluation
train_model()
evaluate_model()

# Plot training data examples
plt.figure(figsize=(10, 4))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(train_data[i].numpy(), cmap="gray")
    plt.title(f"Label: {train_labels[i].item()}")
    plt.axis("off")
plt.tight_layout()
plt.show()
