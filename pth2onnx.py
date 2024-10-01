import torch
import onnx

# Load your PyTorch model (.pth file)
model = torch.load('your_model.pth')
model.eval()

# Create dummy input based on the input shape of your model
dummy_input = torch.randn(1, 3, 224, 224)  # Example for an image model

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, "model.onnx", export_params=True)
