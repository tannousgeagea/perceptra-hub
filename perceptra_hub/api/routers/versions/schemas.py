
from uuid import UUID
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ============= Schemas =============

class VersionCreate(BaseModel):
    version_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    storage_profile_id: Optional[UUID] = None
    export_format: str = Field(default="yolo", pattern="^(yolo|coco|pascal_voc|tfrecord|custom)$")
    export_config: dict = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "version_name": "v1.0-release",
                "description": "Initial training dataset",
                "export_format": "yolo",
                "export_config": {
                    "image_size": 640,
                    "include_augmentation": False
                }
            }
        }


class VersionUpdate(BaseModel):
    version_name: Optional[str] = None
    description: Optional[str] = None
    export_config: Optional[dict] = None


class VersionImageAdd(BaseModel):
    project_image_ids: List[int] = Field(..., min_items=1)
    split: str = Field(default="train", pattern="^(train|val|test)$")
    
    class Config:
        json_schema_extra = {
            "example": {
                "project_image_ids": [1, 2, 3, 4, 5],
                "split": "train"
            }
        }


class VersionResponse(BaseModel):
    id: str
    version_id: str
    version_name: str
    version_number: int
    description: Optional[str]
    export_format: str
    export_status: str
    total_images: int
    total_annotations: int
    train_count: int
    val_count: int
    test_count: int
    file_size: Optional[int]
    is_ready: bool
    created_at: str
    exported_at: Optional[str]
    created_by: Optional[str]


class VersionStatistics(BaseModel):
    total_images: int
    total_annotations: int
    splits: dict
    class_distribution: dict
    average_annotations_per_image: float

class ExportConfigRequest(BaseModel):
    """Export configuration request."""
    image_size: Optional[int] = Field(None, ge=128, le=4096, description="Resize images to square")
    image_quality: int = Field(95, ge=1, le=100, description="JPEG quality")
    normalize: bool = Field(False, description="Normalize coordinates to 0-1")
    include_difficult: bool = Field(True, description="Include difficult annotations")
    include_predictions: bool = Field(False, description="Include model predictions")
    min_annotation_area: float = Field(0.0, ge=0.0, le=1.0, description="Filter annotations smaller than this")
    
    # Augmentation
    augment: bool = Field(False, description="Apply data augmentation")
    augmentation_factor: int = Field(1, ge=1, le=10, description="Number of augmented copies per image")
    augmentation_config: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "horizontal_flip": True,
            "vertical_flip": False,
            "rotation_limit": 15,
            "brightness_contrast": True,
            "blur": False,
            "noise": False
        },
        description="Augmentation parameters"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_size": 640,
                "image_quality": 95,
                "augment": True,
                "augmentation_factor": 2,
                "augmentation_config": {
                    "horizontal_flip": True,
                    "rotation_limit": 15,
                    "brightness_contrast": True
                }
            }
        }