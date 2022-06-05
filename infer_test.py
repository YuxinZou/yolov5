import torch

# Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
model = torch.hub.load('.', 'custom', path='./weights/yolov5s.pt', source='local')
# Images
img = './data/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
for i in range(10):
    results = model(img)

print(results)
# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
