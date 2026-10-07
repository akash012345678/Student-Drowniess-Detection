"""
Head Pose Estimation Module using OpenCV solvePnP.
Computes 3D head orientation (Pitch, Yaw, Roll) from 2D facial landmarks.
Classifies head direction (Looking Left, Right, Up, Down, Normal).
"""
import cv2
import numpy as np
from src.config import POSE_2D_LANDMARKS, POSE_3D_MODEL_POINTS, SETTINGS
from utils.logger import logger


class HeadPoseEstimator:
    """Estimates head pose orientation (Pitch, Yaw, Roll) using Perspective-n-Point (solvePnP)."""

    def __init__(self):
        self.model_points = POSE_3D_MODEL_POINTS
        self.indices = POSE_2D_LANDMARKS

    def estimate_pose(
        self, landmarks_pixel: np.ndarray, frame_shape: tuple[int, int]
    ) -> dict:
        """
        Estimate 3D head pose from pixel landmarks and frame dimensions.

        Args:
            landmarks_pixel (np.ndarray): Pixel coordinate landmarks (N, 3)
            frame_shape (tuple): (height, width)

        Returns:
            dict containing:
                - pitch (float): Pitch angle in degrees (positive = up, negative = down)
                - yaw (float): Yaw angle in degrees (positive = right, negative = left)
                - roll (float): Roll angle in degrees (positive = tilt right)
                - pose_state (str): 'Looking Left', 'Looking Right', 'Looking Up', 'Looking Down', 'Normal'
                - rotation_vector (np.ndarray)
                - translation_vector (np.ndarray)
        """
        default_res = {
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "pose_state": "Normal",
            "rotation_vector": None,
            "translation_vector": None,
        }

        if landmarks_pixel is None or len(landmarks_pixel) == 0:
            return default_res

        h, w = frame_shape[:2]

        try:
            # Extract key 2D points for solvePnP
            image_points = np.array(
                [landmarks_pixel[idx][:2] for idx in self.indices], dtype=np.float64
            )

            # Camera matrix estimation
            focal_length = w
            center = (w / 2.0, h / 2.0)
            camera_matrix = np.array(
                [
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )

            # Assuming zero lens distortion
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rvec, tvec = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if not success:
                return default_res

            # Convert rotation vector to rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)

            # Decompose rotation matrix into Euler angles
            # Combine camera matrix and rotation matrix
            proj_matrix = np.hstack((rmat, tvec))
            euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

            pitch = float(euler_angles[0, 0])
            yaw = float(euler_angles[1, 0])
            roll = float(euler_angles[2, 0])

            # Pitch, Yaw, Roll normalization adjustment for standard face orientation
            pitch = ((pitch + 180) % 360) - 180
            yaw = ((yaw + 180) % 360) - 180

            # Roll normalization centered around 0 degrees for level head
            roll = ((roll + 180) % 360) - 180
            if roll > 90.0:
                roll -= 180.0
            elif roll < -90.0:
                roll += 180.0

            # Determine pose state based on thresholds
            pose_state = self.classify_pose_state(pitch, yaw)

            return {
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "roll": round(roll, 2),
                "pose_state": pose_state,
                "rotation_vector": rvec,
                "translation_vector": tvec,
            }

        except Exception as e:
            logger.error(f"Head pose estimation error: {e}")
            return default_res

    def classify_pose_state(self, pitch: float, yaw: float) -> str:
        """Classify head orientation direction based on pitch and yaw angles."""
        yaw_right = SETTINGS["thresholds"]["yaw_right_threshold"]
        yaw_left = SETTINGS["thresholds"]["yaw_left_threshold"]
        pitch_up = SETTINGS["thresholds"]["pitch_up_threshold"]
        pitch_down = SETTINGS["thresholds"]["pitch_down_threshold"]

        if yaw < yaw_left:
            return "Looking Left"
        elif yaw > yaw_right:
            return "Looking Right"
        elif pitch > pitch_down:
            return "Looking Down"
        elif pitch < pitch_up:
            return "Looking Up"
        else:
            return "Normal"


head_pose_estimator = HeadPoseEstimator()
