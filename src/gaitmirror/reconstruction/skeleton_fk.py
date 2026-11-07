"""
Kinematic Skeleton với Forward Kinematics (FK) cho bài toán LLD.

Bộ xương tham số hóa phục vụ tối ưu hóa 3D trong pipeline gait_mirror.
Tuân thủ ràng buộc LLD: 4 biến bone lengths chi dưới hoàn toàn độc lập.

Phả hệ xương (parent-child) theo chuẩn SMPL-compatible 17 khớp:
  - Root: pelvis (0)
  - Trái: l_hip (1) -> l_knee (4) -> l_ankle (7) -> l_foot (10)
  - Phải: r_hip (2) -> r_knee (5) -> r_ankle (8) -> r_foot (11)
  - Spine: spine (3) -> chest (6) -> neck (9) -> head (12)
  - Arms: l_shoulder (13) -> l_elbow (15) -> l_wrist (17)
          r_shoulder (14) -> r_elbow (16) -> r_wrist (18)

LLD constraint: femur_l, femur_r, tibia_l, tibia_r là 4 biến riêng biệt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from gaitmirror.reconstruction.mirror_optimization import DecoupledLegLengths


# ---------------------------------------------------------------------------
# Định nghĩa cấu trúc xương
# ---------------------------------------------------------------------------

@dataclass
class BoneDefinition:
    """Định nghĩa một đoạn xương trong cây phả hệ."""
    name: str
    parent_idx: int      # Index khớp cha (-1 nếu là root)
    child_idx: int       # Index khớp con
    length_mm: float     # Độ dài xương (mm)
    # Trục cục bộ của xương trong hệ tọa độ cha (đơn vị)
    local_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))


# ---------------------------------------------------------------------------
# Kinematic Skeleton
# ---------------------------------------------------------------------------

class KinematicSkeleton:
    """
    Bộ xương động học tham số hóa với FK.

    Hỗ trợ bài toán LLD thông qua DecoupledLegLengths:
      - bone_length_femur_l và bone_length_femur_r là 2 biến độc lập.
      - bone_length_tibia_l và bone_length_tibia_r là 2 biến độc lập.
      - Tuyệt đối không ép L_femur_left == L_femur_right trong quá trình tối ưu.

    Attributes:
        leg_lengths: Instance DecoupledLegLengths giữ 4 tham số LLD.
        joint_positions: (J, 3) vị trí các khớp trong không gian 3D.
    """

    # Tên khớp chuẩn (17 khớp body)
    JOINT_NAMES: List[str] = [
        "pelvis",          # 0 - root
        "l_hip",           # 1
        "r_hip",           # 2
        "spine",           # 3
        "l_knee",          # 4
        "r_knee",          # 5
        "chest",           # 6
        "l_ankle",         # 7
        "r_ankle",         # 8
        "neck",            # 9
        "l_foot",          # 10
        "r_foot",          # 11
        "head",            # 12
        "l_shoulder",      # 13
        "r_shoulder",      # 14
        "l_elbow",         # 15
        "r_elbow",         # 16
        "l_wrist",         # 17 (optional)
        "r_wrist",         # 18 (optional)
    ]

    # Phả hệ: (child_idx, parent_idx)
    PARENTS: Dict[int, int] = {
        0: -1,   # pelvis = root
        1: 0,    # l_hip <- pelvis
        2: 0,    # r_hip <- pelvis
        3: 0,    # spine <- pelvis
        4: 1,    # l_knee <- l_hip
        5: 2,    # r_knee <- r_hip
        6: 3,    # chest <- spine
        7: 4,    # l_ankle <- l_knee
        8: 5,    # r_ankle <- r_knee
        9: 6,    # neck <- chest
        10: 7,   # l_foot <- l_ankle
        11: 8,   # r_foot <- r_ankle
        12: 9,   # head <- neck
        13: 9,   # l_shoulder <- neck
        14: 9,   # r_shoulder <- neck
        15: 13,  # l_elbow <- l_shoulder
        16: 14,  # r_elbow <- r_shoulder
        17: 15,  # l_wrist <- l_elbow
        18: 16,  # r_wrist <- r_elbow
    }

    def __init__(
        self,
        leg_lengths: Optional[DecoupledLegLengths] = None,
        spine_length_mm: float = 300.0,
        upper_arm_length_mm: float = 280.0,
        forearm_length_mm: float = 250.0,
        hip_width_mm: float = 180.0,
        shoulder_width_mm: float = 360.0,
        foot_length_mm: float = 50.0,
    ) -> None:
        """
        Khởi tạo bộ xương với các tham số chiều dài.

        Args:
            leg_lengths: DecoupledLegLengths với 4 tham số LLD độc lập.
                Nếu None, tạo mặc định symmetric (femur=400, tibia=350).
            spine_length_mm: Chiều dài cột sống (mm).
            upper_arm_length_mm: Chiều dài cánh tay trên (mm).
            forearm_length_mm: Chiều dài cẳng tay (mm).
            hip_width_mm: Nửa độ rộng hông (mm), dùng cho offset l_hip/r_hip.
            shoulder_width_mm: Nửa độ rộng vai (mm).
            foot_length_mm: Chiều dài bàn chân (mm).
        """
        self.leg_lengths = leg_lengths if leg_lengths is not None else DecoupledLegLengths()
        self.spine_length_mm = float(spine_length_mm)
        self.upper_arm_length_mm = float(upper_arm_length_mm)
        self.forearm_length_mm = float(forearm_length_mm)
        self.hip_width_mm = float(hip_width_mm)
        self.shoulder_width_mm = float(shoulder_width_mm)
        self.foot_length_mm = float(foot_length_mm)

        # Vị trí khớp trong T-pose (J, 3), khởi tạo bằng forward kinematics
        self._joint_positions: np.ndarray = self._compute_tpose()

    def _compute_tpose(self) -> np.ndarray:
        """
        Tính vị trí T-pose của tất cả các khớp.

        Hệ tọa độ: Y-up, Z-forward (OpenCV camera space không dùng ở đây —
        đây là skeleton space để tối ưu hóa).

        Returns:
            positions: (J, 3) float64, trong đó J = len(JOINT_NAMES).
        """
        J = len(self.JOINT_NAMES)
        pos = np.zeros((J, 3), dtype=np.float64)

        ll = self.leg_lengths  # Viết tắt

        # Root: pelvis tại gốc tọa độ
        pos[0] = [0.0, 0.0, 0.0]

        # Hông: offset ngang ±hip_width, cùng mức pelvis
        pos[1] = [-self.hip_width_mm, 0.0, 0.0]   # l_hip (trái = -x)
        pos[2] = [+self.hip_width_mm, 0.0, 0.0]   # r_hip (phải = +x)

        # Spine: lên trên
        pos[3] = [0.0, self.spine_length_mm * 0.4, 0.0]   # spine (40% cột sống)
        pos[6] = [0.0, self.spine_length_mm * 0.75, 0.0]  # chest
        pos[9] = [0.0, self.spine_length_mm, 0.0]          # neck
        pos[12] = [0.0, self.spine_length_mm + 120.0, 0.0] # head

        # --- CHI DƯỚI TRÁI (độc lập theo LLD) ---
        # l_knee = l_hip xuống femur_l
        pos[4] = pos[1] + [0.0, -ll.bone_length_femur_l, 0.0]
        # l_ankle = l_knee xuống tibia_l
        pos[7] = pos[4] + [0.0, -ll.bone_length_tibia_l, 0.0]
        # l_foot
        pos[10] = pos[7] + [0.0, -self.foot_length_mm, 0.0]

        # --- CHI DƯỚI PHẢI (độc lập theo LLD) ---
        # r_knee = r_hip xuống femur_r
        pos[5] = pos[2] + [0.0, -ll.bone_length_femur_r, 0.0]
        # r_ankle = r_knee xuống tibia_r
        pos[8] = pos[5] + [0.0, -ll.bone_length_tibia_r, 0.0]
        # r_foot
        pos[11] = pos[8] + [0.0, -self.foot_length_mm, 0.0]

        # Vai: offset ngang ±shoulder_width từ neck
        pos[13] = pos[9] + [-self.shoulder_width_mm, 0.0, 0.0]  # l_shoulder
        pos[14] = pos[9] + [+self.shoulder_width_mm, 0.0, 0.0]  # r_shoulder

        # Cánh tay trái
        pos[15] = pos[13] + [-self.upper_arm_length_mm, 0.0, 0.0]  # l_elbow
        pos[17] = pos[15] + [-self.forearm_length_mm, 0.0, 0.0]    # l_wrist

        # Cánh tay phải
        pos[16] = pos[14] + [+self.upper_arm_length_mm, 0.0, 0.0]  # r_elbow
        pos[18] = pos[16] + [+self.forearm_length_mm, 0.0, 0.0]    # r_wrist

        return pos

    @property
    def joint_positions(self) -> np.ndarray:
        """Vị trí T-pose các khớp (J, 3)."""
        return self._joint_positions

    def get_bone_lengths_lower_limb(self) -> np.ndarray:
        """
        Trả về 4 tham số bone lengths chi dưới dạng mảng.
        Thứ tự: [femur_l, femur_r, tibia_l, tibia_r].

        Dùng để truyền vào optimizer như 4 biến độc lập.
        """
        return self.leg_lengths.as_array()

    def set_bone_lengths_lower_limb(self, params: np.ndarray) -> None:
        """
        Cập nhật 4 tham số từ optimizer [femur_l, femur_r, tibia_l, tibia_r].
        Sau đó tính lại T-pose.
        """
        self.leg_lengths.update_from_array(params)
        self._joint_positions = self._compute_tpose()

    def compute_lld_mm(self) -> Tuple[float, float, float]:
        """
        Tính chênh lệch chiều dài chi.

        Returns:
            (lld_femur_mm, lld_tibia_mm, lld_total_mm)
            Dương = chân trái dài hơn chân phải.
        """
        ll = self.leg_lengths
        return ll.lld_femur_mm, ll.lld_tibia_mm, ll.lld_total_mm

    def __repr__(self) -> str:
        lld_f, lld_t, lld_tot = self.compute_lld_mm()
        return (
            f"KinematicSkeleton("
            f"femur=({self.leg_lengths.bone_length_femur_l:.0f}/{self.leg_lengths.bone_length_femur_r:.0f})mm, "
            f"tibia=({self.leg_lengths.bone_length_tibia_l:.0f}/{self.leg_lengths.bone_length_tibia_r:.0f})mm | "
            f"LLD={lld_tot:+.1f}mm)"
        )

