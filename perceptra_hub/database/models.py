from django.db import models

# Create your models here.
# Create your models here.
# class ProjectType(models.Model):
#     project_type = models.CharField(max_length=255)
#     description = models.CharField(max_length=255)
#     created_at = models.DateTimeField(auto_now_add=True)
#     meta_info = models.JSONField(null=True, blank=True)
    
#     class Meta:
#         db_table = 'project_type'
#         verbose_name_plural = 'Project Type'
        
#     def __str__(self):
#         return self.project_type
    
# class ImageMode(models.Model):
#     mode = models.CharField(max_length=255)
#     description = models.CharField(max_length=255)
#     created_at = models.DateTimeField(auto_now_add=True)
#     meta_info = models.JSONField(null=True, blank=True)
    
#     class Meta:
#         db_table = 'image_mode'
#         verbose_name_plural = 'Image Mode'
        
#     def __str__(self):
#         return self.mode
    
# class Project(models.Model):
#     project_id = models.CharField(max_length=255)
#     project_name = models.CharField(max_length=255)
#     project_type = models.ForeignKey(ProjectType, on_delete=models.CASCADE)
#     annotation_group = models.CharField(max_length=255)
#     description = models.CharField(max_length=255)
#     created_at = models.DateTimeField(auto_now_add=True)
#     meta_info = models.JSONField(null=True, blank=True)
    
#     class Meta:
#         db_table = 'project'
#         verbose_name_plural = 'Project'
        
#     def __str__(self):
#         return f'{self.project_name} - {self.project_type}'
    
# class Annotation(models.Model):
#     project = models.ForeignKey(Project, on_delete=models.CASCADE)
#     image = models.ForeignKey(Image, on_delete=models.CASCADE)
#     annotation_file = models.FileField(upload_to='labels/')
#     created_at = models.DateTimeField(auto_now_add=True)
#     meta_info = models.JSONField(null=True, blank=True)
    
#     class Meta:
#         db_table = 'annotation'
#         verbose_name_plural = 'Annotations'
        
#     def __str__(self):
#         return f'annotation for {self.image.image_name} at {self.project.project_name}'