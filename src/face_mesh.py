"""
MediaPipe FaceMesh Detection & Tracking Module.
Handles multi-face detection, selects the primary (largest) face,
and extracts normalized 3D and pixel 2D facial landmarks.
"""
import cv2
import warnings

# Suppress benign protobuf deprecation warnings triggered by MediaPipe internals
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import mediapipe as mp
import numpy as np
from utils.logger import logger


class FaceMeshDetector:
    """Wrapper for MediaPipe FaceMesh with multi-face tracking and landmark extraction."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 5,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process frame and return landmark data for the largest face detected.

        Returns:
            dict containing:
                - face_found (bool): True if at least one face detected
                - multiple_faces (bool): True if more than 1 face detected
                - face_count (int): Total faces detected
                - landmarks_norm (np.ndarray): Shape (N, 3) normalized landmarks (0.0 to 1.0)
                - landmarks_pixel (np.ndarray): Shape (N, 3) pixel coordinate landmarks
                - bbox (tuple): (x_min, y_min, width, height) of primary face
        """
        result = {
            "face_found": False,
            "multiple_faces": False,
            "face_count": 0,
            "landmarks_norm": None,
            "landmarks_pixel": None,
            "bbox": None,
        }

        if frame is None or frame.size == 0:
            return result

        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            mesh_results = self.face_mesh.process(rgb_frame)
            if not mesh_results.multi_face_landmarks:
                return result

            face_count = len(mesh_results.multi_face_landmarks)
            result["face_count"] = face_count
            result["multiple_faces"] = face_count > 1

            # Select largest face by bounding box area
            largest_face = None
            max_area = -1
            best_bbox = None
            best_norm_landmarks = None
            best_pixel_landmarks = None

            for face_landmarks in mesh_results.multi_face_landmarks:
                # Convert landmarks to numpy array
                norm_coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark],
                    dtype=np.float64,
                )
                pixel_coords = np.array(
                    [[lm.x * w, lm.y * h, lm.z * w] for lm in face_landmarks.landmark],
                    dtype=np.float64,
                )

                x_min, y_min = np.min(pixel_coords[:, :2], axis=0)
                x_max, y_max = np.max(pixel_coords[:, :2], axis=0)
                box_w = max(0, x_max - x_min)
                box_h = max(0, y_max - y_min)
                area = box_w * box_h

                if area > max_area:
                    max_area = area
                    best_bbox = (int(x_min), int(y_min), int(box_w), int(box_h))
                    best_norm_landmarks = norm_coords
                    best_pixel_landmarks = pixel_coords

            if best_norm_landmarks is not None:
                result["face_found"] = True
                result["landmarks_norm"] = best_norm_landmarks
                result["landmarks_pixel"] = best_pixel_landmarks
                result["bbox"] = best_bbox

        except Exception as e:
            logger.error(f"FaceMesh processing error: {e}")

        return result

    def close(self):
        """Release FaceMesh resources."""
        self.face_mesh.close()
