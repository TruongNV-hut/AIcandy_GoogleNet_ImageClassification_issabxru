"""

@author:  AIcandy 
@website: aicandy.vn

"""

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
from aicandy_model_src_igrxgxxe.aicandy_googlenet_model_ealuvpor import GoogleNet
import os


# python aicandy_googlenet_train_odiiiyry.py --train_dir ../dataset --num_epochs 10 --batch_size 32 --model_path aicandy_model_out_bretqhex/aicandy_model_pth_syliacip.pth

def train(train_dir, num_epochs, batch_size, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training with: ", device)

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    
    # Split dataset into training and validation sets
    dataset_size = len(dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Save class labels
    with open('label.txt', 'w') as f:
        for idx, label in enumerate(dataset.classes):
            f.write(f"{idx}: {label}\n")

    num_classes = len(dataset.classes)
    model = GoogleNet(num_classes=num_classes).to(device)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_dataset)
        train_acc = 100 * correct / total
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_dataset)
        val_acc = 100 * val_correct / val_total

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f'Saved best model with accuracy: {best_acc:.2f}%')

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='AIcandy.vn')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to save the best model')

    args = parser.parse_args()
    train(args.train_dir, args.num_epochs, args.batch_size, args.model_path)
