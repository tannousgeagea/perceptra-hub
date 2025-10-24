import os
import cv2
import time
import random
import numpy as np
from fastapi import APIRouter, HTTPException, status
from datetime import datetime
from typing import Callable, Optional
from fastapi import Request
from fastapi import Response
from django.db.models import F
from fastapi.routing import APIRoute, APIRouter
from ml_models.models import ModelVersion
from annotations.models import Annotation, AnnotationGroup, AnnotationClass
from inferences.models import PredictionImageResult, PredictionOverlay
from common_utils.metrics.utils import annotation_to_box, ap_per_class
from common_utils.detection.utils import box_iou_batch

class TimedRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        async def custom_route_handler(request: Request) -> Response:
            before = time.time()
            response: Response = await original_route_handler(request)
            duration = time.time() - before
            response.headers["X-Response-Time"] = str(duration)
            print(f"route duration: {duration}")
            print(f"route response: {response}")
            print(f"route response headers: {response.headers}")
            return response

        return custom_route_handler
    
router = APIRouter(
    route_class=TimedRoute,
    responses={404: {"description": "Not found"}},
)

@router.api_route(
    "/validation/metrics/{model_version_id}", methods=["GET"], tags=["Validation"]
    )
def fetch_validation_metrics(
    model_version_id:int
):
    try:
        model_version = ModelVersion.objects.get(id=model_version_id)
    except ModelVersion.DoesNotExist:
        raise HTTPException(status_code=404, detail="ModelVersion not found")


    image_results = PredictionImageResult.objects.filter(model_version_id=model_version_id)

    project = model_version.model.project
    annotation_group = AnnotationGroup.objects.filter(project=project).first()
    classes = annotation_group.classes.all().order_by('class_id')

    pred_boxes = []
    gt_boxes = []

    all_tp, all_conf, all_pred_cls, all_target_cls = [], [], [], []
    iou_thresholds = np.linspace(0.5, 0.95, 10)  # T = 10 thresholds
    T = len(iou_thresholds)
    
    for res in image_results[4:]:
        pred_boxes = np.array([
            annotation_to_box(
                xyxy=p.bbox,
                class_id=p.class_id if p.class_id is not None else classes.filter(name=p.class_label.lower()).first().class_id,
                conf=p.confidence,
                image_id=res.id,
            ) for p in res.overlays.all()
        ])

        annots = Annotation.objects.filter(
            project_image__image=res.image,
            is_active=True
        )

        gt_boxes = np.array([
            annotation_to_box(
                xyxy=ann.data,
                class_id=ann.annotation_class.class_id,
                image_id=res.id,
            ) for ann in annots
        ])

        boxes_true = gt_boxes[:, 2:6] if len(gt_boxes) else np.empty((0, 4))
        boxes_pred = pred_boxes[:, 2:6] if len(pred_boxes) else np.empty((0, 4))

        # Compute IoUs only if there is at least one prediction and one GT
        if len(boxes_true) and len(boxes_pred):
            ious = box_iou_batch(boxes_true, boxes_pred)
        else:
            ious = np.zeros((len(boxes_true), len(boxes_pred)))

        Np, Ng = len(pred_boxes), len(gt_boxes)
        if Np == 0:
            continue

        tps = np.zeros((Np, T), dtype=bool)
        conf = pred_boxes[:, 6]
        pred_cls = pred_boxes[:, 1]

        for t_idx, iou_thr in enumerate(iou_thresholds):
            matched_gt = set()
            for pred_idx in np.argsort(-conf):  # Highest confidence first
                cls = pred_cls[pred_idx]
                candidates = [
                    gt_idx for gt_idx in range(Ng)
                    if int(gt_boxes[gt_idx, 1]) == int(cls) and gt_idx not in matched_gt
                ]

                if not candidates:
                    continue

                ious_for_pred = ious[candidates, pred_idx]
                best_idx = np.argmax(ious_for_pred)
                best_gt_idx = candidates[best_idx]
                if ious_for_pred[best_idx] >= iou_thr:
                    tps[pred_idx, t_idx] = True
                    matched_gt.add(best_gt_idx)

        all_tp.append(tps)
        all_conf.append(conf)
        all_pred_cls.append(pred_cls)
        if len(gt_boxes):
            all_target_cls.append(gt_boxes[:, 1])

    tp = np.concatenate(all_tp, axis=0)                  # shape: (N_total_pred, T)
    conf = np.concatenate(all_conf, axis=0)              # shape: (N_total_pred,)
    pred_cls = np.concatenate(all_pred_cls, axis=0)      # shape: (N_total_pred,)
    target_cls = np.concatenate(all_target_cls, axis=0)  # shape: (N_total_gt,)

    class_names = {
        cls.class_id: cls.name for cls in classes
    }

    tp, fp, p, r, f1, ap, unique_classes, p_curve, r_curve, f1_curve, x, prec_values = ap_per_class(
        tp=tp,
        conf=conf,
        pred_cls=pred_cls,
        target_cls=target_cls,
        plot=False,
        names=class_names,
    )

    return {
        "precision": round(float(p.mean(0)), 3),
        "recall": round(float(r.mean(0)), 3),
        "f1": round(float(f1.mean(0)), 3),
        "map": round(float(ap[:, 0].mean()), 3),
        "best_threshold": round(x[f1_curve.mean(0).argmax()], 3),
        "precisionConfidence": [
            {
                "confidence": c, "precision": pc
            } for c, pc in zip(x.tolist(), p_curve.mean(0).tolist())
        ],
        "recallConfidence": [
            {
                "confidence": c, "recall": rc
            } for c, rc in zip(x.tolist(), r_curve.mean(0).tolist())
        ],
        "precisionRecall": [
            {
                "recall": rec, "precision": prec
            } for rec, prec in zip(x.tolist(), prec_values.mean(0).tolist())
        ],
        "f1Confidence": [
            {
                "confidence": c, "f1Score": f1
            } for c, f1 in zip(x.tolist(),f1_curve.mean(0).tolist())
        ],
    }