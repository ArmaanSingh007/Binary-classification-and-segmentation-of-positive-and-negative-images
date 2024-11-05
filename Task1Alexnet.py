import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to check if a file is an image file
def is_image_file(filename):
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    return any(filename.lower().endswith(ext) for ext in extensions)

# Define the dataset class
class CrackDataset(Dataset):
    def __init__(self, positive_dir, negative_dir, transform=None, max_images=80000):
        self.positive_dir = positive_dir
        self.negative_dir = negative_dir
        self.transform = transform
        self.image_list = []
        self.labels = []

        # Load positive images
        for filename in os.listdir(positive_dir):
            if is_image_file(filename):
                self.image_list.append(os.path.join(positive_dir, filename))
                self.labels.append(1)  # Label for positive images

        # Load negative images
        for filename in os.listdir(negative_dir):
            if is_image_file(filename):
                self.image_list.append(os.path.join(negative_dir, filename))
                self.labels.append(0)  # Label for negative images

        # Limit to max_images
        if len(self.image_list) > max_images:
            indices = np.random.choice(len(self.image_list), max_images, replace=False)
            self.image_list = [self.image_list[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

        # Debug statement to check the filtered image files
        if not self.image_list:
            print(f"No image files found in {positive_dir} or {negative_dir}. Please check the directories and file extensions.")
        else:
            print(f"Found {len(self.image_list)} image files in total.")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, img_path

# Define paths with corrected and verified directory paths
train_positive_dir = r'C:\Users\Armaan\OneDrive\Desktop\Projecttask1\Data\training data\test\Positive'
train_negative_dir = r'C:\Users\Armaan\OneDrive\Desktop\Projecttask1\Data\training data\test\Negative'
val_positive_dir = r'C:\Users\Armaan\OneDrive\Desktop\Projecttask1\Data\Valiadation data\valid\Positive'
val_negative_dir = r'C:\Users\Armaan\OneDrive\Desktop\Projecttask1\Data\Valiadation data\valid\Negative'

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Check if directories exist
if not os.path.exists(train_positive_dir) or not os.path.exists(train_negative_dir) or \
   not os.path.exists(val_positive_dir) or not os.path.exists(val_negative_dir):
    raise ValueError("One or more directories do not exist. Please check the paths.")

if __name__ == '__main__':
    # Create datasets and dataloaders with a limit of 'n' images each
    train_dataset = CrackDataset(positive_dir=train_positive_dir, negative_dir=train_negative_dir, transform=transform, max_images=40000)
    val_dataset = CrackDataset(positive_dir=val_positive_dir, negative_dir=val_negative_dir, transform=transform, max_images=40000)

    # Increased batch size and data loading workers
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Define the model with dropout added
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device)
    num_ftrs = model.classifier[6].in_features
    model.classifier[5] = nn.Dropout(0.5)  # Add dropout before the final layer
    model.classifier[6] = nn.Linear(num_ftrs, 2).to(device)  # Modify the classifier layer to match the number of classes

    # Define the loss function and optimizer with a lower learning rate
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lowering the learning rate

    # Enable mixed precision training (if supported)
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    # Training and validation loop with tracking of loss and accuracy
    num_epochs = 10
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    all_train_labels = []
    all_train_preds = []
    all_val_labels = []
    all_val_preds = []

    # Start timing
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Update running loss
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect all labels and predictions for confusion matrix
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(predicted.cpu().numpy())

            
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_images, val_labels, _ in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)
                val_running_loss += val_loss.item()

                # Calculate validation accuracy
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

                # Collect labels and predictions for confusion matrix
                all_val_labels.extend(val_labels.cpu().numpy())
                all_val_preds.extend(val_predicted.cpu().numpy())

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_accuracy = 100 * val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.2f}%")

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Finished Training and Validation in {elapsed_time:.2f} seconds")

    # Plotting training and validation metrics
    epochs = np.arange(1, num_epochs+1)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    plt.show()

    # Confusion matrix for training
    train_cm = confusion_matrix(all_train_labels, all_train_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=train_cm, display_labels=["Negative", "Positive"])
    disp.plot()
    plt.title("Training Confusion Matrix")
    plt.show()

    # Confusion matrix for validation
    val_cm = confusion_matrix(all_val_labels, all_val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=["Negative", "Positive"])
    disp.plot()
    plt.title("Validation Confusion Matrix")
    plt.show()

    # Display some predictions from the training set
    model.eval()
    with torch.no_grad():
        train_images, train_labels, _ = next(iter(train_loader))
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        train_outputs = model(train_images)
        _, train_predicted = torch.max(train_outputs.data, 1)

        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        axs = axs.flatten()

        for i in range(9):
            img = train_images[i].cpu().numpy().transpose((1, 2, 0))
            img = np.clip(img, 0, 1)
            label = train_labels[i].cpu().item()
            pred = train_predicted[i].cpu().item()
            axs[i].imshow(img)
            axs[i].set_title(f"Label: {label}, Pred: {pred}")
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()

    # Display some predictions from the validation set
    with torch.no_grad():
        val_images, val_labels, _ = next(iter(val_loader))
        val_images, val_labels = val_images.to(device), val_labels.to(device)
        val_outputs = model(val_images)
        _, val_predicted = torch.max(val_outputs.data, 1)

        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        axs = axs.flatten()

        for i in range(9):
            img = val_images[i].cpu().numpy().transpose((1, 2, 0))
            img = np.clip(img, 0, 1)
            label = val_labels[i].cpu().item()
            pred = val_predicted[i].cpu().item()
            axs[i].imshow(img)
            axs[i].set_title(f"Label: {label}, Pred: {pred}")
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()


