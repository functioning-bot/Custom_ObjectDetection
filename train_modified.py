#!/usr/bin/env python3
"""
Modified Model Training Script for ResNet-101 + FPN
Deeper backbone compared to baseline ResNet-50
"""

import torch
import os
import sys
import json
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset, DataLoader
import pickle
# Add project root to path
sys.path.append('/content/Custom_ObjectDetection/DeepDataMiningLearning')
from DeepDataMiningLearning.detection.models import create_detectionmodel

class COCODataset(Dataset):
    """COCO format dataset loader with robust path handling"""
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation_file)
        
        # Keep only images that have at least one valid annotation (w>0, h>0)
        all_ids = list(sorted(self.coco.imgs.keys()))
        valid_ids = []
        for img_id in all_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            valid_anns = [a for a in anns if a.get('bbox') and a['bbox'][2] > 0 and a['bbox'][3] > 0]
            if len(valid_anns) > 0:
                valid_ids.append(img_id)
        self.ids = valid_ids
        
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Load image
        img_info = coco.loadImgs(img_id)[0]
        
        # Handle dataset prefix in filename robustly
        file_name = img_info['file_name']
        dataset_prefix = img_info.get('dataset', '')
        basename = os.path.basename(file_name)
        
        # Try multiple possible file locations
        candidates = []
        if dataset_prefix:
            candidates.append(os.path.join(self.root, 'images', f"{dataset_prefix}_{basename}"))
        # Fallbacks
        candidates.append(os.path.join(self.root, 'images', basename))
        candidates.append(os.path.join(self.root, 'images', file_name))
        
        img_path = None
        for cand in candidates:
            if os.path.exists(cand):
                img_path = cand
                break
                
        if img_path is None:
            raise FileNotFoundError(f"Image not found. Tried: {candidates}")
            
        img = Image.open(img_path).convert("RGB")
        
        # Get bounding boxes and labels
        boxes = []
        labels = []
        areas = []
        crowds = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                boxes.append([x, y, x+w, y+h])
                labels.append(ann['category_id'])
                areas.append(ann.get('area', w * h))
                crowds.append(ann.get('iscrowd', 0))
        
        # Ensure at least one box remains
        if len(boxes) == 0:
            raise FileNotFoundError(f"No valid boxes for image_id {img_id} (filtered).")

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([img_id])
        area = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(crowds, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    metric_logger = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    # Add transform to convert PIL Image to tensor and normalize
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
    ])
    
    for images, targets in metric_logger:
        # Convert PIL Images to tensors and move to device
        images = [transform(image).to(device) for image in images]
        
        # Don't stack the images, keep them as a list of tensors
        # The model's forward pass will handle variable-sized images
        
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Calculate losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(losses):
            print(f"Loss is {losses}, stopping training")
            print(loss_dict)
            sys.exit(1)

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.set_postfix(loss=losses.item())

def evaluate(model, data_loader, device, coco_gt):
    model.eval()
    coco_results = []
    
    # Add transform to convert PIL Image to tensor and normalize
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            # Convert PIL Images to tensors and move to device
            images = [transform(img).to(device) for img in images]
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            outputs = model(images)
            
            # Process outputs
            for target, output in zip(targets, outputs):
                if 'boxes' not in output or len(output['boxes']) == 0:
                    continue
                    
                image_id = int(target["image_id"].item())
                boxes = output["boxes"].tolist()
                scores = output["scores"].tolist()
                labels = output["labels"].tolist()
                
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": [x1, y1, w, h],
                        "score": score
                    })
    
    # Evaluate using COCO metrics
    if len(coco_results) > 0:
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0]  # mAP@0.5:0.95
    else:
        return 0.0
def evaluate_final(model, data_loader, device, coco_gt, checkpoint_path=None):
    """Run final evaluation on the test set"""
    if checkpoint_path:
        try:
            # First try with weights_only=True (safer)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except (pickle.UnpicklingError, RuntimeError) as e:
            print(f"Warning: Could not load with weights_only=True: {e}")
            print("Trying with weights_only=False (only do this if you trust the source)")
            # Fall back to weights_only=False if needed
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load model state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If it's a direct state dict
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {checkpoint_path}")
    
    model.eval()
    coco_results = []
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Final Evaluation"):
            # Convert PIL Images to tensors and move to device
            images = [transform(img).to(device) for img in images]
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            outputs = model(images)
            
            # Process outputs
            for target, output in zip(targets, outputs):
                if 'boxes' not in output or len(output['boxes']) == 0:
                    continue
                    
                image_id = int(target["image_id"].item())
                boxes = output["boxes"].tolist()
                scores = output["scores"].tolist()
                labels = output["labels"].tolist()
                
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": [x1, y1, w, h],
                        "score": score
                    })
    
    # Evaluate using COCO metrics
    if len(coco_results) > 0:
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        
        # Run evaluation
        print("\n" + "="*50)
        print("Final Evaluation Results")
        print("="*50)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Print AP for each category
        print("\nAP per category:")
        for idx, cat_id in enumerate(coco_gt.getCatIds()):
            cat_name = coco_gt.loadCats(cat_id)[0]['name']
            print(f"{cat_name}: {coco_eval.eval['precision'][0, :, idx, 0, -1].mean():.3f}")
        
        return coco_eval.stats
    else:
        print("No detections to evaluate!")
        return None
def evaluate_fast(model, data_loader, device, coco_gt, num_samples=100):
    """Faster evaluation using a subset of the test data"""
    model.eval()
    coco_results = []
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(data_loader, desc="Fast Evaluation")):
            if i * data_loader.batch_size >= num_samples:
                break
                
            # Convert PIL Images to tensors and move to device
            images = [transform(img).to(device) for img in images]
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            outputs = model(images)
            
            # Process outputs
            for target, output in zip(targets, outputs):
                if 'boxes' not in output or len(output['boxes']) == 0:
                    continue
                    
                image_id = int(target["image_id"].item())
                boxes = output["boxes"].tolist()
                scores = output["scores"].tolist()
                labels = output["labels"].tolist()
                
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": [x1, y1, w, h],
                        "score": score
                    })
    
    # Evaluate using COCO metrics
    if len(coco_results) > 0:
        coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        
        # Run evaluation
        print("\n" + "="*50)
        print(f"Fast Evaluation Results (using {len(coco_results)} detections from {num_samples} samples)")
        print("="*50)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Print AP for each category
        print("\nAP per category:")
        for idx, cat_id in enumerate(coco_gt.getCatIds()):
            cat_name = coco_gt.loadCats(cat_id)[0]['name']
            print(f"{cat_name}: {coco_eval.eval['precision'][0, :, idx, 0, -1].mean():.3f}")
        
        return coco_eval.stats
    else:
        print("No detections to evaluate!")
        return None

def main():
    # Configuration
    DATA_PATH = '/content/drive/MyDrive/CMPE_Output/merged_dataset'  # Updated to use Drive directly
    ANNOTATION_FILE = f'{DATA_PATH}/annotations.json'
    MODEL_NAME = 'customrcnn_resnet101'
    NUM_CLASSES = 5  # 4 classes + background
    BATCH_SIZE = 4
    NUM_EPOCHS = 7
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    TRAINABLE_LAYERS = 3 
    CHECKPOINT_DIR = '/content/drive/MyDrive/CMPE_Output/checkpoints/modified'
    SUBSET_SIZE = int(os.environ.get('SUBSET_SIZE', '0'))  # 0 means use full dataset
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Create model with ResNet-101 + FPN
    print("Creating model with ResNet-101 + FPN...")
    model, _, _ = create_detectionmodel(
    modelname=MODEL_NAME,
    num_classes=NUM_CLASSES,
    trainable_layers=TRAINABLE_LAYERS,
    device=str(device)
)
    model = model.to(device) 
    
    # Dataset and DataLoader
    print(f"Loading dataset from {DATA_PATH}...")
    dataset = COCODataset(DATA_PATH, ANNOTATION_FILE)
    
    # Optional: limit dataset to a small random subset for quick sanity checks
    if SUBSET_SIZE > 0:
        k = min(SUBSET_SIZE, len(dataset))
        random.seed(42)
        subset_ids = random.sample(dataset.ids, k)
        dataset.ids = subset_ids
        print(f"[Subset] Using {len(dataset)} images out of original dataset")
    
    # Split into train/val
    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(0.8 * len(indices))
    train_dataset = torch.utils.data.Subset(dataset, indices[:train_size])
    val_dataset = torch.utils.data.Subset(dataset, indices[train_size:])
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, 
        num_workers=4, collate_fn=collate_fn
    )
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LEARNING_RATE, 
        momentum=MOMENTUM, 
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.1
    )
    
    # Load COCO for evaluation
    coco_gt = COCO(ANNOTATION_FILE)
    
    # Training loop
    best_map = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 10)
        
        # Train for one epoch
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        if (epoch + 1) % 2 == 0:  # Evaluate every 2 epochs
            print("\nEvaluating...")
            map_score = evaluate(model, val_loader, device, coco_gt)
            print(f"mAP@[0.5:0.95] = {map_score:.4f}")
            
            # Save best model
            if map_score > best_map:
                best_map = map_score
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'map': map_score,
                }, checkpoint_path)
                print(f"Saved best model with mAP: {best_map:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map': map_score,
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1}")
    # Create test dataset and loader (use your validation set if no test set)
    test_dataset = COCODataset(DATA_PATH, ANNOTATION_FILE)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=2, 
        shuffle=False,
        num_workers=2, 
        collate_fn=collate_fn
    )

    # Load COCO for evaluation
    coco_gt = COCO(ANNOTATION_FILE)

    # Run final evaluation on the best model
    best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    stats = evaluate_fast(model, test_loader, device, coco_gt, num_samples=100)
    
    if stats is not None:
        print(f"\nFast mAP@0.5:0.95 = {stats[0]:.4f}")
        print(f"mAP@0.5 = {stats[1]:.4f}")
        print(f"mAP@0.75 = {stats[2]:.4f}")
    # stats = evaluate_final(model, test_loader, device, coco_gt, checkpoint_path=best_model_path)
    
    # # Print mAP
    # if stats is not None:
    #     print(f"\nFinal mAP@0.5:0.95 = {stats[0]:.4f}")
    #     print(f"mAP@0.5 = {stats[1]:.4f}")
    #     print(f"mAP@0.75 = {stats[2]:.4f}")
if __name__ == "__main__":
    import math  # For math.isfinite check
    main()