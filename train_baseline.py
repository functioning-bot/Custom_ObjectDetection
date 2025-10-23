#!/usr/bin/env python3
"""
Baseline Training Script for Faster R-CNN with ResNet-50 + FPN
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class COCODataset(Dataset):
    """COCO format dataset loader"""
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Load image
        img_info = coco.loadImgs(img_id)[0]
        
        # Handle dataset prefix in filename
        file_name = img_info['file_name']
        dataset_prefix = img_info.get('dataset', '')
        if dataset_prefix:
            img_path = os.path.join(self.root, 'images', f"{dataset_prefix}_{file_name}")
        else:
            img_path = os.path.join(self.root, 'images', file_name)
        
        img = Image.open(img_path).convert("RGB")
        
        # Get bounding boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(ann['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([img_id])
        area = torch.as_tensor([ann['area'] for ann in anns], dtype=torch.float32)
        iscrowd = torch.as_tensor([ann.get('iscrowd', 0) for ann in anns], dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms:
            img = self.transforms(img)
        else:
            img = torchvision.transforms.ToTensor()(img)
        
        return img, target
    
    def __len__(self):
        return len(self.ids)

def get_model(num_classes):
    """Load Faster R-CNN model with ResNet-50 backbone"""
    # Load pretrained model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def collate_fn(batch):
    """Custom collate function for dataloader"""
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    """Train for one epoch"""
    model.train()
    losses = []
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    for i, (images, targets) in enumerate(pbar):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses_batch = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses_batch.backward()
        optimizer.step()
        
        losses.append(losses_batch.item())
        
        if i % print_freq == 0:
            pbar.set_postfix({
                'loss': f"{losses_batch.item():.4f}",
                'avg_loss': f"{np.mean(losses):.4f}"
            })
    
    return np.mean(losses)

@torch.no_grad()
def evaluate(model, data_loader, device, coco_gt):
    """Evaluate model using COCO metrics"""
    model.eval()
    
    coco_results = []
    
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = list(img.to(device) for img in images)
        
        outputs = model(images)
        
        for target, output in zip(targets, outputs):
            image_id = target['image_id'].item()
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                coco_results.append({
                    'image_id': image_id,
                    'category_id': int(label),
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    'score': float(score)
                })
    
    # Evaluate
    if len(coco_results) > 0:
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        return coco_eval.stats[0]  # mAP@0.5:0.95
    else:
        return 0.0

def main():
    # Configuration
    DATA_PATH = '/content/drive/MyDrive/CMPE_Dataset/merged_dataset'
    ANNOTATION_FILE = f'{DATA_PATH}/annotations.json'
    NUM_CLASSES = 5  # 4 classes + background
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    CHECKPOINT_DIR = '/content/drive/MyDrive/CMPE_Output/checkpoints/baseline'
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = COCODataset(DATA_PATH, ANNOTATION_FILE)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )
    
    # Model
    print("Creating model...")
    model = get_model(NUM_CLASSES)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LEARNING_RATE,
        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.1
    )
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_maps = []
    best_map = 0.0
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch
        )
        train_losses.append(train_loss)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate every 2 epochs
        if (epoch + 1) % 2 == 0:
            print(f"\nEvaluating epoch {epoch}...")
            val_map = evaluate(model, val_loader, device, dataset.coco)
            val_maps.append(val_map)
            print(f"Validation mAP: {val_map:.4f}")
            
            # Save best model
            if val_map > best_map:
                best_map = val_map
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'map': val_map,
                }, f'{CHECKPOINT_DIR}/best_model.pth')
                print(f"✓ Saved best model (mAP: {val_map:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'{CHECKPOINT_DIR}/checkpoint_epoch_{epoch}.pth')
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_map = evaluate(model, val_loader, device, dataset.coco)
    print(f"Final mAP: {final_map:.4f}")
    print(f"Best mAP: {best_map:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    eval_epochs = list(range(1, NUM_EPOCHS, 2))
    plt.plot(eval_epochs, val_maps, marker='o')
    plt.title('Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP@0.5:0.95')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{CHECKPOINT_DIR}/training_curves.png')
    plt.show()
    
    print(f"\n✓ Training complete! Best model saved to: {CHECKPOINT_DIR}/best_model.pth")

if __name__ == "__main__":
    main()
