from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
from ultralytics import YOLOWorld

class Detector_rn50():
    def __init__(self):
        self.processor = DetrImageProcessor.from_pretrained("./detr/facebook/detr-resnet-50", revision="no_timm", cache_dir="./detr/")
        self.model = DetrForObjectDetection.from_pretrained("./detr/facebook/detr-resnet-50", revision="no_timm", cache_dir="./detr/")
        
    def detect_and_crop(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')

        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

        crops = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                    f"Detected {self.model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
            )

            a = image.crop(box)
            crops.append(a)
            a.save(f"logs/crop{box}.jpeg")
            
        return crops

class Detector():
    def __init__(self):
        self.model = YOLOWorld("yolov8x-worldv2.pt")  # or select yolov8m/l-world.pt for different sizes
        
    def detect_and_crop(self, image):
        # Detect and crop regions of interests
        bboxes = self.model.predict(source=image, save=True, conf=0.1)[0].boxes.xyxy
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        crops = []
        for box in bboxes:
            box = [round(i, 2) for i in box.tolist()]
            a = image.crop(box)
            crops.append(a)
            
        return crops

if __name__ == "__main__":
    dect = Detector()
    image = Image.open("data/MyVLM/bull/bull_2.jpg")
    dect.detect_and_crop(image)