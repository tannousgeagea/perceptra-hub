import os
import uuid
import django
import shutil
django.setup()
import numpy as np

from annotations.models import (
    Annotation
)



def xyxy2xywh(xyxy):
    """
    Convert bounding box coordinates from (xmin, ymin, xmax, ymax) format to (x, y, width, height) format.

    Parameters:
    - xyxy (Tuple[int, int, int, int]): A tuple representing the bounding box coordinates in (xmin, ymin, xmax, ymax) format.

    Returns:
    - Tuple[int, int, int, int]: A tuple representing the bounding box in (x, y, width, height) format. 
                                 (x, y) are  the center of the bounding box.
    """
    xmin, ymin, xmax, ymax = xyxy
    w = xmax - xmin
    h = ymax - ymin
    return (xmin + w/2, ymin + h/2, w, h)

def save_annotations_into_txtfile(annotations: Annotation, filename:str, dest:str):
    success = False
    try:
        file_name = f"{filename}.txt" 
        label_location = os.path.join(dest, "labels")
        os.makedirs(label_location, exist_ok=True)
        
        if not annotations.exists():
            with open(label_location + "/" + file_name, "w") as file:
                file.writelines([])
                return True
        
        data = [
            [ann.annotation_class.class_id] + list(xyxy2xywh(ann.data)) for ann in annotations
        ]
        
        lines = (("%g " * len(line)).rstrip() % tuple(line) + "\n" for line in data)
        with open(label_location + "/" + file_name, "w") as file:
            file.writelines(lines)
        success = True
            
    except Exception as err:
        raise ValueError(f"Error in saving annotation into txtfile: {err}")
    
    return success

def format_annotation(annotation, format="yolo"):
    """Format annotation based on YOLO, COCO, or Pascal VOC formats."""
    data = annotation.data
    
    if format == "yolo":
        x_min, y_min, x_max, y_max = data 
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        return f"{annotation.annotation_class.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

    elif format == "coco":
        # COCO format JSON
        return {
            "image_id": annotation.project_image.image.image_name,
            "category_id": annotation.annotation_class.class_id,
            "bbox": data['bbox'],
            "segmentation": data.get("segmentation", []),
            "area": (data['bbox'][2] - data['bbox'][0]) * (data['bbox'][3] - data['bbox'][1]),
            "iscrowd": 0
        }

    elif format == "pascal":
        # Pascal VOC XML format
        return f"<object><name>{annotation.annotation_class.name}</name><bndbox><xmin>{data['bbox'][0]}</xmin><ymin>{data['bbox'][1]}</ymin><xmax>{data['bbox'][2]}</xmax><ymax>{data['bbox'][3]}</ymax></bndbox></object>\n"

    return ""

def read_annotation(bbox:list, label:int, image_name:str=None, format="yolo"):
    """Format annotation based on YOLO, COCO, or Pascal VOC formats."""
    
    if format == "yolo":
        x_min, y_min, x_max, y_max = bbox 
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        return f"{int(label)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

    elif format == "coco":
        # COCO format JSON
        return {
            "image_id": image_name,
            "category_id": label,
            "bbox": bbox,
            "segmentation": bbox.get("segmentation", []),
            "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
            "iscrowd": 0
        }

    elif format == "pascal":
        # Pascal VOC XML format
        return f"<object><name>{label}</name><bndbox><xmin>{bbox[0]}</xmin><ymin>{bbox[1]}</ymin><xmax>{bbox[2]}</xmax><ymax>{bbox[3]}</ymax></bndbox></object>\n"

    return ""