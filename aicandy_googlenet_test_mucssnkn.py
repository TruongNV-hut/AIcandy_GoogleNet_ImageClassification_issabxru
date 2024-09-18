"""

@author:  AIcandy 
@website: aicandy.vn

"""

import torch
import torchvision.transforms as transforms
from PIL import Image
from aicandy_model_src_igrxgxxe.aicandy_googlenet_model_ealuvpor import GoogleNet

# python aicandy_googlenet_test_mucssnkn.py --image_path ../image_test.jpg --model_path aicandy_model_out_bretqhex/aicandy_model_pth_syliacip.pth --label_path label.txt

def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = {int(line.split(": ")[0]): line.split(": ")[1].strip() for line in f}
    print('labels: ',labels)
    return labels

def predict(image_path, model_path, labels_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = load_labels(label_path)
    num_classes = len(labels)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    model = GoogleNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    predicted_class = labels[predicted.item()]    
    return labels.get(predicted_class, "Unknown")
    

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='AIcandy.vn')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--label_path', type=str, required=True, help='Path to the label file')

    args = parser.parse_args()
    predicted_class = predict(args.image_path, args.model_path, args.label_path)
    print(f'Predicted class: {predicted_class}')
