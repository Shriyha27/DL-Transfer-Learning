# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.


## Neural Network Model
<img width="987" height="792" alt="image" src="https://github.com/user-attachments/assets/1e412a42-ced2-4b84-87ca-f2fd46388fed" />

## DESIGN STEPS
### STEP 1: 
Import required libraries and define image transforms.

### STEP 2: 
Load training and testing datasets using ImageFolder.
### STEP 3: 
Visualize sample images from the dataset.
### STEP 4: 
Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.
### STEP 5: 
Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.
### STEP 6: 
Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions. 





## PROGRAM

### Name: V. Shriyha

### Register Number: 212224230267

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


dataset_path = "C:\\Users\\admin\\Downloads\\chip_data\\dataset"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}\\train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}\\test", transform=transform)

def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images):
        img, label = dataset[i]
        img = img.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
        axes[i].imshow(img)
        axes[i].set_title(f"{dataset.classes[label]}")
        axes[i].axis('off')
    plt.show()

print(f"Number of training samples: {len(train_dataset)}")
first_image, first_label = train_dataset[0]
print(f"First image shape: {first_image.shape}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)
print(f"Number of classes: {num_classes}")

model=models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchsummary import summary
summary(model, (3, 224, 224))

def train_model(model, train_loader, test_loader, num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() 

        train_losses.append(running_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(test_loader))
        model.train()
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

   
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

train_model(model, train_loader,test_loader)
test_model(model, test_loader)

def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted = predicted.item()


    class_names = class_names = dataset.classes
   
    image_to_display = transforms.ToPILImage()(image)
    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_display)
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted]}')

predict_image(model, image_index=55, dataset=test_dataset)

predict_image(model, image_index=25, dataset=test_dataset)

```

### OUTPUT
<img width="700" height="705" alt="image" src="https://github.com/user-attachments/assets/61e4554d-1e99-4e9a-8b26-b5fdb031e733" />

## Training Loss, Validation Loss Vs Iteration Plot

<img width="995" height="977" alt="image" src="https://github.com/user-attachments/assets/f412e8b5-cd5e-4a3c-b8c3-96e344d01773" />


## Confusion Matrix
<img width="866" height="779" alt="image" src="https://github.com/user-attachments/assets/0922202a-d6c2-45b2-8d33-ec1d64764d9c" />

## Classification Report
<img width="560" height="266" alt="image" src="https://github.com/user-attachments/assets/2e5d3cf8-02a0-4c26-8153-b303febf8d4d" />

### New Sample Data Prediction
<img width="678" height="614" alt="image" src="https://github.com/user-attachments/assets/42e68f5c-3bde-4d41-ba72-adf2b4cca8fa" />
<img width="632" height="610" alt="image" src="https://github.com/user-attachments/assets/b24d90c4-aa20-4090-b9d5-3ddd06579df6" />


## RESULT
Thus the python program to develop an image classification model using transfer learning with VGG19 architecture is executed successfully.
