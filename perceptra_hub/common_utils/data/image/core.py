
import io
import os
import uuid
import shutil
import django
django.setup()
from pathlib import Path
from fastapi import UploadFile
from images.models import Image
from tenants.models import (
    SensorBox,
)

from PIL import Image as PILImage
from projects.models import Project, ProjectImage
from django.core.files.base import ContentFile
from common_utils.data.integrity import validate_image_exists

def compress_image(file, quality:int=65):
    image_file = PILImage.open(file.file)
    img_size = image_file.size
    compressed_io = io.BytesIO()
    image_file.convert("RGB").save(
        compressed_io,
        format="JPEG",
        optimize=True,
        quality=quality
    )

    compressed_io.seek(0)
    return compressed_io.read(), img_size

def register_image_into_db(file, image_id:str=None, source=None, meta_info:dict=None):
    success = False
    result = ''
    try:
        file_ext = f".{file.filename.split('.')[-1]}"
        filename = file.filename.split(file_ext)[0]
        if validate_image_exists(filename=filename):
            image = Image.objects.filter(image_name=filename).first()
            result = {
                'filename': file.filename,
                'status': 'failed',
                'reason': 'Image already exists',
                'image_id': image.image_id,
            }
            
            return success, result, image
        
        file_content, img_size = compress_image(file=file)
        image = Image(
            image_name=filename,
            image_id=image_id if image_id else str(uuid.uuid4()),
            source_of_origin=source,
            meta_info=meta_info,
            sensorbox=SensorBox.objects.filter(sensor_box_name=source).first()
        )
        
        image.width, image.height = img_size
        image.image_file.save(
            os.path.basename(file.filename), 
            ContentFile(file_content)
            )
        image.save()
        success = True
        result = 'success'
        return success, result, image
    except Exception as err:
        raise ValueError(f'failed to register image into db: {err}')
    
    
def save_image_file(file_path:str, file:UploadFile):
    success = False
    try:
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        
        file_path = Path(file_path + "/" + file.filename)
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)   
        
        success = True
    except Exception as err:
        raise ValueError(f'failed to save image file in {file_path}: {err}')
    
    return success 

def save_image(file, image_id:str=None, project_id=None, source=None, meta_info:dict=None):
    success = False
    try:
        success, result, image = register_image_into_db(
            file=file,
            image_id=image_id,
            source=source,
            meta_info=meta_info,
        ) 

        # if not success:
        #     return success, result

        print(result)
        if project_id:
            project = Project.objects.filter(
                name=project_id
            ).first()

            if not project:
                result = {
                    'filename': file.filename,
                    'status': 'failed',
                    'reason': f"Project {project_id} not found",
                    'image_id': image.image_id,
                }
                return success, result
            
            ProjectImage.objects.get_or_create(
                project=project,
                image=image
            )

        result = {
            'filename': file.filename,
            'status': 'success',
            'image_id': image.image_id,
            }
        
        success = True
        
    except Exception as err:
        result = {
            'filename': file.filename,
            'status': 'failed',
            'reason': str(err)
        }
        
    return success, result

