"""
Keypoint definitions and symmetry flip mappings for ViTPose WholeBody (133 keypoints).
Compatible with COCO-WholeBody topology.
"""

from typing import List, Tuple
import numpy as np


# -------------------------------------------------------------------------
# 1. COCO-WholeBody 133 Keypoint Index References
# -------------------------------------------------------------------------
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Foot keypoints (Critical for Zeni Heel-Strike & Toe-Off events)
L_BIG_TOE, L_SMALL_TOE, L_HEEL = 17, 18, 19
R_BIG_TOE, R_SMALL_TOE, R_HEEL = 20, 21, 22

# -------------------------------------------------------------------------
# 2. Symmetry Pairs for Left <-> Right Swap (Mirror Reflection)
# -------------------------------------------------------------------------
WHOLEBODY_FLIP_PAIRS: List[Tuple[int, int]] = [
    # Body (COCO 17)
    (L_EYE, R_EYE),
    (L_EAR, R_EAR),
    (L_SHOULDER, R_SHOULDER),
    (L_ELBOW, R_ELBOW),
    (L_WRIST, R_WRIST),
    (L_HIP, R_HIP),
    (L_KNEE, R_KNEE),
    (L_ANKLE, R_ANKLE),
    
    # Feet (6 points)
    (L_BIG_TOE, R_BIG_TOE),
    (L_SMALL_TOE, R_SMALL_TOE),
    (L_HEEL, R_HEEL),
    
    # Face (68 points: 23-90)
    # Jawline (23-39, center chin is 31)
    *[(23 + i, 39 - i) for i in range(8)],
    # Eyebrows (40-49)
    *[(40 + i, 49 - i) for i in range(5)],
    # Nose nostrils (54-58)
    (54, 58), (55, 57),
    # Eyes (59-70)
    *[(59 + i, 65 + i) for i in range(6)],
    # Lips (71-90)
    (71, 77), (72, 76), (73, 75),
    (78, 82), (79, 81),
    (83, 87), (84, 86),
    (88, 90),
    
    # Hands (Each hand 21 points: Left 91-111, Right 112-132)
    *[(91 + i, 112 + i) for i in range(21)]
]


def compute_neck_and_root(kpts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính tọa độ điểm Cổ (Neck) và Hông chậu (Root/Pelvis) từ ViTPose WholeBody.
    - Neck: Trung điểm 2 vai (L_SHOULDER, R_SHOULDER)
    - Root: Trung điểm 2 háng (L_HIP, R_HIP)
    
    Args:
        kpts: Mảng (133, 2) hoặc (133, 3).
    Returns:
        neck_xy (2,), root_xy (2,)
    """
    neck = (kpts[L_SHOULDER, :2] + kpts[R_SHOULDER, :2]) / 2.0
    root = (kpts[L_HIP, :2] + kpts[R_HIP, :2]) / 2.0
    return neck, root