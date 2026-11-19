from fastapi import APIRouter, Depends, HTTPException
from users.auth import get_current_user
from augmentations.models import Augmentation, AugmentationParameterAssignment
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

# Pydantic Models for Response
class AugmentationParameterResponse(BaseModel):
    name: str
    parameter_type: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_value: Optional[float] = None

class AugmentationResponse(BaseModel):
    id: int
    name: str
    title: str
    description: str
    thumbnail: str
    parameters: List[AugmentationParameterResponse]


DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

router = APIRouter(
    responses={404: {"description": "Not found"}},
)

@router.api_route(
    "/augmentations", methods=["GET"], tags=["Augmentations"], response_model=List[AugmentationResponse]
    )
def get_active_augmentations(_user=Depends(get_current_user)):
    """
    Retrieve all active augmentations with their associated parameters.
    """
    try:
        augmentations = Augmentation.objects.filter(is_active=True).prefetch_related('assigned_parameters__parameter')

        augmentation_list = []
        for augmentation in augmentations:
            parameters = [
                AugmentationParameterResponse(
                    name=assignment.parameter.name,
                    parameter_type=assignment.parameter.parameter_type,
                    min_value=assignment.min_value,
                    max_value=assignment.max_value,
                    default_value=assignment.default_value
                )
                for assignment in AugmentationParameterAssignment.objects.filter(augmentation=augmentation)
            ]

            augmentation_list.append(AugmentationResponse(
                id=augmentation.id,
                name=augmentation.name,
                title=augmentation.title,
                thumbnail=augmentation.thumbnail.url if augmentation.thumbnail else "",
                description=augmentation.description,
                parameters=parameters
            ))

        return augmentation_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching augmentations: {str(e)}")
