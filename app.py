from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load your PyTorch model (.pth)
model = torch.load('your_model.pth')
model.eval()

# Define image transformation (adjust based on your model input)
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize image
    transforms.ToTensor(),           # Convert image to Tensor
])

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get image from the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    img = Image.open(file)
    
    # Preprocess the image
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
    
    # Assuming the output is a classification model (adjust based on your model type)
    predicted_class = torch.argmax(output, dim=1).item()

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)