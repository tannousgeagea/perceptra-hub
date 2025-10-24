import cv2
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image as PILImage

def save_image(output_dir, file_name_prefix, image, mask=None, quality:int=65):
    """
    Save the augmented image.

    Args:
        output_dir (str): Output directory.
        file_name_prefix (str): File name prefix.
        image (numpy.ndarray): Image to save.
        mask (numpy.ndarray, optional): Mask to save.
    """
    image_path = Path(output_dir) / f"{file_name_prefix}.jpg"
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(image_rgb)

    pil_img.save(
        str(image_path),
        format="JPEG",
        quality=quality,
        optimize=True
    )

    if mask is not None:
        mask_path = Path(output_dir) / f"{file_name_prefix}_mask.png"
        cv2.imwrite(str(mask_path), mask)

    return str(image_path)


def save_annotations(output_dir, file_name_prefix, bboxes, labels, annotation_type="detection"):
    """
    Save the annotations to a JSON file.

    Args:
        output_dir (str): Output directory.
        file_name_prefix (str): File name prefix.
        bboxes (list): List of bounding boxes.
        labels (list): List of labels for bounding boxes.
        annotation_type (str): Type of annotation.
    """
    annotations = {
        "type": annotation_type,
        "bboxes": bboxes,
        "labels": labels,
    }
    annotation_path = Path(output_dir) / f"{file_name_prefix}.json"
    with open(annotation_path, "w") as f:
        json.dump(annotations, f, indent=4)

    return str(annotation_path)

def decode_image(file_bytes):
    """Convert file bytes into a cv2 image (BGR)"""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img