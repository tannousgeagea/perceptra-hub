
import os
import shutil
import django
django.setup()
import numpy as np
from pathlib import Path
from fastapi import UploadFile
from database.models import Annotation

def get_class_id_from_file(file):
    try:
        assert os.path.exists(file) ,f'File Not Found {file}'
        with open(file) as f:
            lb = [x.split()[0] for x in f.read().strip().splitlines() if len(x)]
            lb = np.array(lb, dtype=np.int8)
            lb = np.unique(lb).tolist()
    
    except Exception as err:
        raise ValueError(f'failed to get class_id from file: {err}')
    
    return lb

def load_xyxy_from_file(file):
    """
    Extract bounding box data from a text file.

    This function reads a text file containing bounding box data. Each line in the file should 
    represent a bounding box or a polygon, starting with a class ID followed by the vertices coordinates.
    If a line contains more than 4 coordinates, it is treated as a polygon and converted to an axis-aligned
    bounding box. The function returns class IDs and bounding boxes.

    Parameters:
    - txt_file (str): The path to the text file containing the bounding box data.

    Returns:
    - A tuple of two lists, the first being class IDs and 
      the second being bounding boxes (each box either as (xmin, ymin, xmax, ymax) or as a polygon)
    """
    assert os.path.exists(file) ,f'File Not Found {file}'
    
    with open(file) as f:
        lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
        lb = np.array(lb, dtype=np.float32)
        
    nl = len(lb)
    if nl:
        assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
        assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
        assert (lb[:, 1:] <= 1).all(), f"non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}"
        _, i = np.unique(lb, axis=0, return_index=True)
        if len(i) < nl:  # duplicate row check
            lb = lb[i]  # remove duplicates
            msg = f"WARNING ⚠️ {file}: {nl - len(i)} duplicate labels removed"
            print(msg)
    else:
        lb = np.zeros((0, 5), dtype=np.float32)
    
    return lb

def save_annotation_file(file_path:str, file:UploadFile):
    success = False
    try:
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        
        file_path = Path(file_path + "/" + file.filename)
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    
        success = True
    except Exception as err:
        raise ValueError(f'failed to save annotation file: {err}')

    return success


def save_annotation_raw_into_txtfile(file_path:str, filename:str, data:list):
    success = False
    try:
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        
        file_path = Path(file_path + "/" + filename)
        with file_path.open("wb") as f:
            f.writelines(data)
    
        success = True
    except Exception as err:
        raise ValueError(f'failed to save raw annotation file into txt file: {err}')

    return success


def register_annotation_into_db(image, project, annotation_file, meta_info):
    success = False
    try:
        annotation = Annotation()
        annotation.image = image
        annotation.project = project
        annotation.annotation_file = annotation_file
        annotation.meta_info = meta_info
        annotation.save()
        
        image.annotated = True
        image.save()
        success = True
    except Exception as err:
        raise ValueError(f'failed to register annotation in DB: {err}')

    return success

def load_labels(file:str):
    """
    Extract data from a text file.

    This function reads a text file containing bounding box data. Each line in the file should 
    represent a bounding box or a polygon, starting with a class ID followed by the vertices coordinates.
    If a line contains more than 4 coordinates, it is treated as a polygon and converted to an axis-aligned
    bounding box. The function returns class IDs and bounding boxes.

    Parameters:
    - txt_file (str): The path to the text file containing the bounding box data.

    Returns:
    - A tuple of two lists, the first being class IDs and 
      the second being bounding boxes (each box either as (xmin, ymin, xmax, ymax) or as a polygon)
    """
    assert os.path.exists(file) ,f'File Not Found {file}'
    
    with open(file) as f:
        lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
        
    return lb
    