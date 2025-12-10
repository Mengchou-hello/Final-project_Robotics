import onnx
import onnxruntime as ort
import numpy as np

# 1. Load the model
model_path = "best.onnx"
print(f"\n🔍 Loading ONNX model: {model_path}")
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
print("✔ Model is valid ONNX format")

# 2. Create ONNX Runtime session
session = ort.InferenceSession(model_path)
print("✔ ONNX Runtime session created")

# 3. Print input/output layer info
print("\n--- Model Inputs ---")
for i in session.get_inputs():
    print(f"Name: {i.name}, Shape: {i.shape}, Type: {i.type}")

print("\n--- Model Outputs ---")
for o in session.get_outputs():
    print(f"Name: {o.name}, Shape: {o.shape}, Type: {o.type}")
