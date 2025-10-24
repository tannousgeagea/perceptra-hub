from fastapi import APIRouter, Depends
from users.models import CustomUser as User
from organizations.models import Organization
from memberships.models import OrganizationMembership
from jobs.models import Job, JobImage
from pydantic import BaseModel
from typing import Optional
from fastapi import Path
from api.routers.auth.queries.dependencies import (
    organization_access_dependency
)

router = APIRouter()

# class OrgMemberOut(BaseModel):
#     id: int
#     username: str
#     email: str
#     role: str

@router.get("/organizations/{org_id}/progress")
def fetch_user_progress(
    org_id: int,
    _membership=Depends(organization_access_dependency)
):
    memberships = OrganizationMembership.objects.filter(
        organization_id=org_id
    ).select_related("user", "role")


    data = []
    for m in memberships:
        jobs = Job.objects.filter(assignee=m.user).order_by('created_at')
        if not jobs:
            continue
        
        assigned_jobs = []
        image_count, annotated_images, reviewed_images, completed_images = 0, 0, 0, 0
        for job in jobs:
            project_images = JobImage.objects.filter(job=job, project_image__is_active=True)
            image_count += project_images.count()
            annotated_images += project_images.filter(project_image__status="annotated").count()
            reviewed_images += project_images.filter(project_image__status="reviewed").count()
            completed_images += project_images.filter(project_image__status="dataset").count()
            assigned_jobs.append(
                {
                    "jobId": f"{job.id}",
                    "jobName": f"{job.name}",
                    "totalImages": project_images.count(),
                    "annotatedImages": project_images.filter(project_image__status="annotated").count(),
                    "reviewedImages": project_images.filter(project_image__status="reviewed").count(),
                    "completedImages": project_images.filter(project_image__status="dataset").count(),
                }
            )
        
        data.append({
            "userId": str(m.user.id),
            "userName": f"{m.user.username}",
            "userRole": f"{m.role.name}",
            "role": m.role.name,
            "avatarUrl": "undefined",
            "totalImages": image_count,
            "annotatedImages": annotated_images,
            "reviewedImages": reviewed_images,
            "completedImages": completed_images,
            "progressPercentage": round((completed_images / image_count) * 100, 2),
            "lastUpdated": jobs.last().updated_at.strftime("%Y-%m-%d %H:%M:%S"),
            "assignedJobs": assigned_jobs

        })

    return data

"""
      userId: "user-1",
      userName: "Alice Johnson",
      userRole: "Senior Annotator",
      avatarUrl: undefined,
      totalImages: 425,
      annotatedImages: 320,
      reviewedImages: 280,
      completedImages: 250,
      progressPercentage: 75.3,
      lastUpdated: new Date("2024-06-04T14:30:00Z"),
"""