from flask import Flask, render_template, request, redirect, url_for, flash
from torchvision import models, transforms
from PIL import Image
import torch
import os

app = Flask(__name__)
app.secret_key = 'idk'

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

def classify_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

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
        class_id = classify_image(image_path)
        return render_template('result.html', class_id=class_id, image_url=image_path)

if __name__ == '__main__':
    app.run(debug=True)