from transformers import AutoImageProcessor, SwinForImageClassification
from PIL import Image
import torch
image = Image.open(r"C:\Users\S.Sanjaikumar\CODE\MiniProjects\TB TO 3D Model\data\tb_xray.jpg").convert("RGB")
model_id = "gianlab/swin-tiny-patch4-window7-224-finetuned-lungs-disease"
processor = AutoImageProcessor.from_pretrained(model_id)
model = SwinForImageClassification.from_pretrained(model_id)
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
label = model.config.id2label[predicted_class_idx]
print(f"Prediction: {label}")

