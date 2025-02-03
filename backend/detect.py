import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import json
import sys
from engine import train_one_epoch, evaluate
import utils

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class DeepFashionDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        # Load annotations
        annotation_file = os.path.join(root, "annotations.json")
        with open(annotation_file) as f:
            self.annotations = json.load(f)

        self.image_ids = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Get image ID and its annotations
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.root, "images", img_id)
        img = Image.open(img_path).convert("RGB")

        boxes = torch.tensor(self.annotations[img_id]["boxes"], dtype=torch.float32)
        labels = torch.tensor(self.annotations[img_id]["labels"], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target

from torchvision.transforms import functional as F

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            image = F.hflip(image)
            target["boxes"][:, [0, 2]] = 1 - target["boxes"][:, [2, 0]]
        return image, target
def get_model(num_classes):
    # Load Faster R-CNN pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the classifier head with one for your dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
# Set paths
train_dir = "dataset/train"
val_dir = "dataset/val"

# Prepare datasets and data loaders
train_dataset = DeepFashionDataset(train_dir, transforms=Compose([ToTensor(), RandomHorizontalFlip(0.5)]))
val_dataset = DeepFashionDataset(val_dir, transforms=Compose([ToTensor()]))

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=utils.collate_fn)

# Load model
num_classes = 4  # For example, 3 clothing categories + 1 background
model = get_model(num_classes)

# Optimizer and learning rate scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Train the model
num_epochs = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, val_loader, device=device)

# Save the model
torch.save(model.state_dict(), "faster_rcnn_deepfashion.pth")
# Load Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def detect_top_bottom(image_path):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Transform to tensor

    # Perform detection
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    # Extract top and bottom bounding boxes (labels assumed as 1 for top and 2 for bottom)
    results = []
    for i, score in enumerate(predictions['scores']):
        if score > 0.7:  # Confidence threshold
            label = predictions['labels'][i].item()
            bbox = predictions['boxes'][i].tolist()
            results.append({'label': label, 'bbox': bbox, 'score': score.item()})

    return results

if __name__ == "__main__":
    # Receive the image path as an argument from Node.js
    image_path = sys.argv[1]
    detection_results = detect_top_bottom(image_path)
    print(json.dumps(detection_results))  # Output results as JSON
