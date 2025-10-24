from typing import List, Literal, Tuple, TypedDict

class Box(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float
    class_label: str
    confidence: float
    image_id: int

MatchType = Literal["TP", "FP", "FN"]

class MatchResult(TypedDict):
    iou: float
    match_type: MatchType
    prediction: Box
    ground_truth: Box | None
