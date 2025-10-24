
import django
django.setup()
from django.conf import settings
from images.models import Image

def validate_image_exists(filename):
    if Image.objects.filter(image_name=filename).exists(): 
        return True
        
    return False

# def validate_annotation_exists(image, project):
#     if Annotation.objects.filter(project=project, image=image).exists():
#         return True

#     return False

# def validate_project_exists(project_name):
#     if Project.objects.filter(project_name=project_name).exists():
#         return True
    
#     return False