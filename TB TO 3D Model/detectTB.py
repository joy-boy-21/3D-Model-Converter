import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
img_path = r"C:\Users\S.Sanjaikumar\CODE\MiniProjects\TB TO 3D Model\data\tb_xray.jpg"
image = Image.open(img_path).convert("RGB")
img_np = np.array(image)
processor = DetrImageProcessor.from_pretrained("shivamjadon/MaskRCNN-TB-Chest-Xray")
model = DetrForObjectDetection.from_pretrained("shivamjadon/MaskRCNN-TB-Chest-Xray")
model.eval()
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    cv2.rectangle(img_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
    cv2.putText(img_np, "TB Area", (int(box[0]), int(box[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
output_path = r"C:\Users\S.Sanjaikumar\CODE\MiniProjects\TB TO 3D Model\data\tb_xray_labeled.jpg"
cv2.imwrite(output_path, img_np)
