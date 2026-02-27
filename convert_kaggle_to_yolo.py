import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path
import random

def parse_xml_annotation(xml_file):
    """Parse XML annotation file and extract bounding box info"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get image dimensions
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    # Get bounding box
    obj = root.find('object')
    if obj is None:
        return None
        
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    
    return {
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax,
        'width': img_width,
        'height': img_height
    }

def convert_to_yolo_format(bbox, img_width, img_height):
    """Convert absolute coordinates to YOLO normalized format"""
    x_center = (bbox['xmin'] + bbox['xmax']) / 2.0
    y_center = (bbox['ymin'] + bbox['ymax']) / 2.0
    width = bbox['xmax'] - bbox['xmin']
    height = bbox['ymax'] - bbox['ymin']
    
    # Normalize to 0-1 range
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm

def main():
    # Paths
    kaggle_dir = Path("car-plate-detection")
    dataset_dir = Path("dataset")
    
    # Create dataset structure
    train_dir = dataset_dir / "images" / "train"
    val_dir = dataset_dir / "images" / "val"
    train_labels_dir = dataset_dir / "labels" / "train"
    val_labels_dir = dataset_dir / "labels" / "val"
    
    for dir_path in [train_dir, val_dir, train_labels_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all annotation files
    annotations_dir = kaggle_dir / "annotations"
    annotation_files = list(annotations_dir.glob("*.xml"))
    
    # Split into train/val (80/20 split)
    random.shuffle(annotation_files)
    split_idx = int(0.8 * len(annotation_files))
    train_files = annotation_files[:split_idx]
    val_files = annotation_files[split_idx:]
    
    print(f"Total files: {len(annotation_files)}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Process training files
    for xml_file in train_files:
        # Copy image
        img_file = kaggle_dir / "images" / f"{xml_file.stem}.png"
        if img_file.exists():
            shutil.copy2(img_file, train_dir / img_file.name)
        
        # Parse and convert annotation
        bbox = parse_xml_annotation(xml_file)
        if bbox:
            x_center, y_center, width, height = convert_to_yolo_format(bbox, bbox['width'], bbox['height'])
            
            # Write YOLO label file
            label_file = train_labels_dir / f"{xml_file.stem}.txt"
            with open(label_file, 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Process validation files
    for xml_file in val_files:
        # Copy image
        img_file = kaggle_dir / "images" / f"{xml_file.stem}.png"
        if img_file.exists():
            shutil.copy2(img_file, val_dir / img_file.name)
        
        # Parse and convert annotation
        bbox = parse_xml_annotation(xml_file)
        if bbox:
            x_center, y_center, width, height = convert_to_yolo_format(bbox, bbox['width'], bbox['height'])
            
            # Write YOLO label file
            label_file = val_labels_dir / f"{xml_file.stem}.txt"
            with open(label_file, 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print("Conversion completed!")
    print(f"Images copied to: {train_dir} and {val_dir}")
    print(f"Labels created in: {train_labels_dir} and {val_labels_dir}")

if __name__ == "__main__":
    main()
