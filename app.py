from flask import Flask, render_template, request, redirect, url_for, flash
from torchvision import models, transforms
from PIL import Image
import torch
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Ensure the upload folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained model
model = models.resnet18(pretrained=True)
model.eval()

# Define image transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet class names
with open('imagenet_classes.txt') as f:
    class_names = [line.strip() for line in f.readlines()]

def classify_image(image_path):
    try:
        image = Image.open(image_path)
        print(f"Image loaded: {image}")
        image = preprocess(image).unsqueeze(0)
        print(f"Preprocessed image: {image.shape}")
        with torch.no_grad():
            output = model(image)
        _, predicted = torch.max(output, 1)
        class_id = predicted.item()
        class_name = class_names[class_id]
        return class_id, class_name
    except Exception as e:
        print(f"Error during image classification: {e}")
        return None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        flash('No image file provided!')
        return redirect(url_for('home'))
    file = request.files['image']
    if file.filename == '':
        flash('No selected file!')
        return redirect(url_for('home'))
    if file:
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)
        class_id, class_name = classify_image(image_path)
        if class_id is None:
            flash('Error during image classification!')
            return redirect(url_for('home'))
        return render_template('result.html', class_id=class_id, class_name=class_name, image_url=image_path)

if __name__ == '__main__':
    app.run(debug=True)
