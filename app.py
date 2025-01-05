import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from PIL import Image
import io

# Define the folder to save uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define model architecture using ResNet50
class ResNetIrisTumor(nn.Module):
    def __init__(self, num_classes, dropout_rate):
        super(ResNetIrisTumor, self).__init__()
        self.resnet = models.resnet50(pretrained=True)  # Use ResNet50 for a deeper model
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
num_classes = 2
dropout_rate = 0.5
model = ResNetIrisTumor(num_classes=num_classes, dropout_rate=dropout_rate)
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')), strict=False)
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define prediction function
def predict_image(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
    return 'Tumor' if predicted.item() == 1 else 'Normal'

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Open and predict
        img = Image.open(filepath)
        prediction = predict_image(img)

        # Redirect to result page with prediction and image path
        return render_template('result.html', prediction=prediction, image_url=url_for('uploaded_file', filename=file.filename))

# Route to serve the uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)


