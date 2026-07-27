"""
src/models/cv_anthropometry.py
──────────────────────────────
Upper-body measurement extraction using MediaPipe Pose Landmarker (Tasks API).

Compatible with mediapipe >= 0.10.0 (Python 3.12 / 3.13 safe).
The old mp.solutions.pose API was removed in 0.10 — this file uses the
replacement mp.tasks.vision.PoseLandmarker API.

OpenCV removed entirely. Image loading uses PIL. MediaPipe receives
a plain RGB numpy array — it has no dependency on OpenCV.

Pipeline:
  1. Download pose_landmarker_heavy.task model on first run (cached to
     artifacts/pose_landmarker.task — ~30 MB, one-time only).
  2. Accept an RGB numpy array (from PIL in main.py).
  3. Detect 33 pose landmarks.
  4. Convert pixel distances to centimetres using a reference object
     or a shoulder-hip heuristic fallback.
  5. Return AnthropometryResult with 4 measurements + confidence.

Expected Input:
  RGB numpy array (H, W, 3) produced by PIL in main.py
  OR a file path string loaded internally via PIL.

Expected Output:
  AnthropometryResult dataclass:
    shoulder_width_cm : float
    chest_width_cm    : float
    torso_length_cm   : float
    arm_length_cm     : float
    confidence        : float  (0-1, landmark visibility average)
"""
from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
from loguru import logger


# ─── Model download ───────────────────────────────────────────────────────────

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)
_MODEL_PATH = Path("artifacts/pose_landmarker.task")


def _ensure_model() -> str:
    """Download the pose landmarker model if not already cached."""
    if not _MODEL_PATH.exists():
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[CVAnthro] pose_landmarker_heavy.task not found — downloading (~30 MB)..."
        )
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        logger.success(f"[CVAnthro] Model saved → {_MODEL_PATH}")
    return str(_MODEL_PATH)


# ─── Data class ───────────────────────────────────────────────────────────────

@dataclass
class AnthropometryResult:
    shoulder_width_cm : float
    chest_width_cm    : float
    torso_length_cm   : float
    arm_length_cm     : float
    confidence        : float

    def to_dict(self) -> dict:
        return {
            "shoulder_width_cm": round(self.shoulder_width_cm, 1),
            "chest_width_cm":    round(self.chest_width_cm, 1),
            "torso_length_cm":   round(self.torso_length_cm, 1),
            "arm_length_cm":     round(self.arm_length_cm, 1),
            "confidence":        round(self.confidence, 3),
        }


# ─── Landmark indices (MediaPipe Pose — 33 keypoints) ─────────────────────────

LM = {
    "nose":       0,
    "l_shoulder": 11,
    "r_shoulder": 12,
    "l_elbow":    13,
    "r_elbow":    14,
    "l_wrist":    15,
    "r_wrist":    16,
    "l_hip":      23,
    "r_hip":      24,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _pixel_dist(a, b, h: int, w: int) -> float:
    """Euclidean pixel distance between two normalised landmarks."""
    ax, ay = a.x * w, a.y * h
    bx, by = b.x * w, b.y * h
    return math.hypot(ax - bx, ay - by)


def _vis(landmark) -> float:
    """Return landmark visibility, defaulting to 0.0 if None."""
    v = landmark.visibility
    return float(v) if v is not None else 0.0


# ─── Main class ───────────────────────────────────────────────────────────────

class CVAnthropometry:
    """
    Upper-body anthropometry via MediaPipe Pose Landmarker (Tasks API).

    Compatible with mediapipe >= 0.10.0.
    No OpenCV dependency — uses PIL for image loading.
    MediaPipe receives a plain RGB numpy array.

    Usage:
        scanner = CVAnthropometry()
        result  = scanner.measure(rgb_array)
        print(result.to_dict())
        scanner.close()

    Context manager:
        with CVAnthropometry() as scanner:
            result = scanner.measure(rgb_array)
    """

    def __init__(
        self,
        reference_width_cm: float = 21.0,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
    ):
        self.reference_width_cm = reference_width_cm

        model_path = _ensure_model()

        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
        logger.info("[CVAnthro] PoseLandmarker initialised (Tasks API).")

    # ── Public API ────────────────────────────────────────────────────────────

    def measure(
        self,
        image_input: str | np.ndarray,
        reference_px: Optional[float] = None,
    ) -> AnthropometryResult:
        """
        Run anthropometry on a single image.

        Args:
            image_input  : File path (str) OR RGB numpy array (H, W, 3).
                           When called from main.py the array comes from
                           PIL Image.open().convert("RGB") → np.array()
                           and is already RGB — no colour conversion needed.
            reference_px : Pixel width of the reference object in the frame.
                           If None, uses shoulder-hip heuristic (~15% less
                           accurate).

        Returns:
            AnthropometryResult
        """
        # ── Load → RGB numpy array ────────────────────────────────────────────
        if isinstance(image_input, str):
            # File path — PIL always returns RGB when convert("RGB") is called
            pil_img = Image.open(image_input).convert("RGB")
            rgb = np.array(pil_img)
        else:
            # Already an RGB numpy array from main.py
            rgb = image_input

        h, w = rgb.shape[:2]

        # ── Run MediaPipe detection ───────────────────────────────────────────
        # mp.Image(SRGB) expects standard RGB — matches what PIL gives us
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results  = self._landmarker.detect(mp_image)

        if not results.pose_landmarks:
            raise ValueError(
                "MediaPipe could not detect a pose in this image. "
                "Ensure the person is clearly visible and front-facing."
            )

        # pose_landmarks is list[list[NormalizedLandmark]] — take first person
        lm = results.pose_landmarks[0]

        # ── Pixels-per-cm scale ───────────────────────────────────────────────
        if reference_px is not None:
            px_per_cm = reference_px / self.reference_width_cm
        else:
            # Heuristic: shoulder → hip vertical distance ≈ 45 cm
            shoulder_hip_px = _pixel_dist(
                lm[LM["l_shoulder"]], lm[LM["l_hip"]], h, w
            )
            px_per_cm = shoulder_hip_px / 45.0
            logger.warning(
                "[CVAnthro] No reference_px supplied — using heuristic scale. "
                "Results may be ~15% off."
            )

        def px2cm(px: float) -> float:
            return px / px_per_cm

        # ── Measurements ──────────────────────────────────────────────────────

        # Shoulder width — landmark 11 (left shoulder) ↔ 12 (right shoulder)
        shoulder_px = _pixel_dist(
            lm[LM["l_shoulder"]], lm[LM["r_shoulder"]], h, w
        )

        # Chest ≈ shoulder × 0.85 (anatomical 2D projection ratio)
        chest_px = shoulder_px * 0.85

        # Torso — midpoint of shoulders → midpoint of hips
        mid_sx = (lm[LM["l_shoulder"]].x + lm[LM["r_shoulder"]].x) / 2 * w
        mid_sy = (lm[LM["l_shoulder"]].y + lm[LM["r_shoulder"]].y) / 2 * h
        mid_hx = (lm[LM["l_hip"]].x    + lm[LM["r_hip"]].x)    / 2 * w
        mid_hy = (lm[LM["l_hip"]].y    + lm[LM["r_hip"]].y)    / 2 * h
        torso_px = math.hypot(mid_sx - mid_hx, mid_sy - mid_hy)

        # Arm — left side: shoulder→elbow + elbow→wrist (two-segment chain)
        upper_arm_px = _pixel_dist(lm[LM["l_shoulder"]], lm[LM["l_elbow"]], h, w)
        lower_arm_px = _pixel_dist(lm[LM["l_elbow"]],   lm[LM["l_wrist"]], h, w)
        arm_px = upper_arm_px + lower_arm_px

        # ── Confidence — average visibility of 6 measurement landmarks ────────
        key_indices = [
            LM["l_shoulder"], LM["r_shoulder"],
            LM["l_hip"],      LM["r_hip"],
            LM["l_elbow"],    LM["l_wrist"],
        ]
        confidence = float(np.mean([_vis(lm[i]) for i in key_indices]))

        return AnthropometryResult(
            shoulder_width_cm=px2cm(shoulder_px),
            chest_width_cm=px2cm(chest_px),
            torso_length_cm=px2cm(torso_px),
            arm_length_cm=px2cm(arm_px),
            confidence=confidence,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self):
        """Release MediaPipe landmarker resources."""
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()