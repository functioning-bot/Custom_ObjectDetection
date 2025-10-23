#!/usr/bin/env python3
"""
Merge Waymo and NuScenes datasets into a unified COCO format
Maps NuScenes categories to Waymo's simpler 4-class scheme
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Category mapping from NuScenes (10 classes) to Waymo (4 classes)
CATEGORY_MAPPING = {
    # NuScenes -> Waymo
    'car': 'Vehicle',
    'truck': 'Vehicle',
    'bus': 'Vehicle',
    'trailer': 'Vehicle',
    'construction_vehicle': 'Vehicle',
    'pedestrian': 'Pedestrian',
    'motorcycle': 'Cyclist',
    'bicycle': 'Cyclist',
    'traffic_cone': 'Sign',
    'barrier': 'Sign'
}

# Unified categories (4 classes + background)
UNIFIED_CATEGORIES = [
    {'id': 1, 'name': 'Vehicle', 'supercategory': 'object'},
    {'id': 2, 'name': 'Pedestrian', 'supercategory': 'object'},
    {'id': 3, 'name': 'Cyclist', 'supercategory': 'object'},
    {'id': 4, 'name': 'Sign', 'supercategory': 'object'}
]

def load_coco_annotation(json_path):
    """Load COCO format annotation file"""
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def create_category_map(categories, mapping=None):
    """Create mapping from original category names to unified IDs"""
    cat_name_to_id = {}
    for cat in categories:
        original_name = cat['name']
        if mapping and original_name in mapping:
            # Map NuScenes categories to Waymo
            unified_name = mapping[original_name]
        else:
            # Keep Waymo categories as-is
            unified_name = original_name
        
        # Find unified category ID
        for unified_cat in UNIFIED_CATEGORIES:
            if unified_cat['name'] == unified_name:
                cat_name_to_id[original_name] = unified_cat['id']
                break
    
    return cat_name_to_id

def _resolve_img_path(base_dir: Path, file_name: str, default_images_subdir: str = 'images') -> Path:
    p = Path(file_name)
    # If file_name already includes a subdirectory like 'images/...', don't append another 'images'
    if len(p.parts) > 1:
        return base_dir / p
    else:
        return base_dir / default_images_subdir / p


def merge_datasets(waymo_path, nuscenes_path, output_path, use_symlinks=False):
    """
    Merge Waymo and NuScenes datasets
    
    Args:
        waymo_path: Path to Waymo dataset folder
        nuscenes_path: Path to NuScenes dataset folder
        output_path: Path to save merged dataset
        use_symlinks: Use symbolic links instead of copying images (faster)
    """
    
    # Create output directories
    output_images = Path(output_path) / 'images'
    output_images.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    waymo_anno = load_coco_annotation(os.path.join(waymo_path, 'annotations.json'))
    nuscenes_anno = load_coco_annotation(os.path.join(nuscenes_path, 'annotations.json'))
    
    # Create category mappings
    waymo_cat_map = create_category_map(waymo_anno['categories'])
    nuscenes_cat_map = create_category_map(nuscenes_anno['categories'], CATEGORY_MAPPING)
    
    print(f"Waymo categories: {waymo_cat_map}")
    print(f"NuScenes categories: {nuscenes_cat_map}")
    
    # Initialize merged dataset
    merged_data = {
        'info': {
            'description': 'Merged Waymo + NuScenes Dataset',
            'version': '1.0',
            'year': 2025,
            'contributor': 'CMPE Assignment',
            'date_created': '2025-10-28'
        },
        'licenses': waymo_anno.get('licenses', []),
        'categories': UNIFIED_CATEGORIES,
        'images': [],
        'annotations': []
    }
    
    image_id_offset = 0
    annotation_id_offset = 0
    
    # Process Waymo dataset
    print("\n=== Processing Waymo Dataset ===")
    waymo_base_path = Path(waymo_path)
    
    for img in tqdm(waymo_anno['images'], desc="Waymo images"):
        new_img_id = img['id'] + image_id_offset
        new_img = img.copy()
        new_img['id'] = new_img_id
        new_img['dataset'] = 'waymo'
        merged_data['images'].append(new_img)
        
        # Copy or symlink image
        src_img = _resolve_img_path(waymo_base_path, img['file_name'], 'images')
        dst_img = output_images / f"waymo_{Path(img['file_name']).name}"
        
        if src_img.exists():
            if use_symlinks:
                if not dst_img.exists():
                    dst_img.symlink_to(src_img)
            else:
                shutil.copy2(src_img, dst_img)
        else:
            print(f"Warning: Image not found: {src_img}")
    
    # Process Waymo annotations
    for ann in tqdm(waymo_anno['annotations'], desc="Waymo annotations"):
        new_ann = ann.copy()
        new_ann['id'] = ann['id'] + annotation_id_offset
        new_ann['image_id'] = ann['image_id'] + image_id_offset
        
        # Map category
        old_cat_id = ann['category_id']
        old_cat_name = next(c['name'] for c in waymo_anno['categories'] if c['id'] == old_cat_id)
        new_ann['category_id'] = waymo_cat_map[old_cat_name]
        
        merged_data['annotations'].append(new_ann)
    
    # Update offsets for NuScenes
    image_id_offset = len(merged_data['images'])
    annotation_id_offset = len(merged_data['annotations'])
    
    # Process NuScenes dataset
    print("\n=== Processing NuScenes Dataset ===")
    nuscenes_base_path = Path(nuscenes_path)
    
    for img in tqdm(nuscenes_anno['images'], desc="NuScenes images"):
        new_img_id = img['id'] + image_id_offset
        new_img = img.copy()
        new_img['id'] = new_img_id
        new_img['dataset'] = 'nuscenes'
        merged_data['images'].append(new_img)
        
        # Copy or symlink image
        src_img = _resolve_img_path(nuscenes_base_path, img['file_name'], 'images')
        dst_img = output_images / f"nuscenes_{Path(img['file_name']).name}"
        
        if src_img.exists():
            if use_symlinks:
                if not dst_img.exists():
                    dst_img.symlink_to(src_img)
            else:
                shutil.copy2(src_img, dst_img)
        else:
            print(f"Warning: Image not found: {src_img}")
    
    # Process NuScenes annotations
    for ann in tqdm(nuscenes_anno['annotations'], desc="NuScenes annotations"):
        new_ann = ann.copy()
        new_ann['id'] = ann['id'] + annotation_id_offset
        new_ann['image_id'] = ann['image_id'] + image_id_offset
        
        # Map category
        old_cat_id = ann['category_id']
        old_cat_name = next(c['name'] for c in nuscenes_anno['categories'] if c['id'] == old_cat_id)
        
        if old_cat_name in nuscenes_cat_map:
            new_ann['category_id'] = nuscenes_cat_map[old_cat_name]
            merged_data['annotations'].append(new_ann)
        else:
            print(f"Warning: Unmapped category: {old_cat_name}")
    
    # Save merged annotations
    output_json = Path(output_path) / 'annotations.json'
    print(f"\n=== Saving merged dataset to {output_json} ===")
    with open(output_json, 'w') as f:
        json.dump(merged_data, f)
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total images: {len(merged_data['images'])}")
    print(f"  - Waymo: {len(waymo_anno['images'])}")
    print(f"  - NuScenes: {len(nuscenes_anno['images'])}")
    print(f"Total annotations: {len(merged_data['annotations'])}")
    print(f"  - Waymo: {len([a for a in waymo_anno['annotations']])}")
    print(f"  - NuScenes: {len([a for a in nuscenes_anno['annotations']])}")
    print(f"\nCategories: {[c['name'] for c in UNIFIED_CATEGORIES]}")
    
    # Count annotations per category
    cat_counts = {cat['name']: 0 for cat in UNIFIED_CATEGORIES}
    for ann in merged_data['annotations']:
        cat_name = next(c['name'] for c in UNIFIED_CATEGORIES if c['id'] == ann['category_id'])
        cat_counts[cat_name] += 1
    
    print("\nAnnotations per category:")
    for cat_name, count in cat_counts.items():
        print(f"  {cat_name}: {count}")
    
    print(f"\n✓ Merged dataset saved to: {output_path}")
    return merged_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge Waymo and NuScenes datasets')
    parser.add_argument('--waymo', required=True, help='Path to Waymo dataset')
    parser.add_argument('--nuscenes', required=True, help='Path to NuScenes dataset')
    parser.add_argument('--output', required=True, help='Output path for merged dataset')
    parser.add_argument('--symlinks', action='store_true', help='Use symlinks instead of copying')
    
    args = parser.parse_args()
    
    merge_datasets(args.waymo, args.nuscenes, args.output, args.symlinks)
