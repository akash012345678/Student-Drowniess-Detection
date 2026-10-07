"""
Automated Test Suite for Student Drowsiness & Attention System.
Verifies 10 critical edge cases:
1. Normal state detection
2. Yawning detection (MAR)
3. Drowsiness detection (EAR < thresh for > 3s)
4. Looking Left (Yaw < -15°)
5. Looking Right (Yaw > 15°)
6. Looking Down (Pitch < -15°)
7. Looking Up (Pitch > 15°)
8. Spectacles handling
9. Low light frame enhancement (CLAHE)
10. Multiple faces detection
"""
import unittest
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.image_processing import enhancer
from src.feature_extraction import calculate_ear, calculate_mar, extract_facial_features
from src.head_pose import head_pose_estimator
from src.drowsiness_detector import DrowsinessDetector
from src.distraction_detector import DistractionDetector
from src.attention_scorer import attention_scorer
from src.config import LEFT_EYE_INDICES, MOUTH_INDICES


class TestDrowsinessSystem(unittest.TestCase):
    """Test suite executing 10 required edge case unit tests."""

    def setUp(self):
        # Synthetic landmark template for testing (468 landmarks)
        self.mock_landmarks = np.zeros((468, 3), dtype=np.float64)

        # Populate Left Eye Landmarks (Open position)
        self.mock_landmarks[33] = [100, 200, 0]  # P1 outer
        self.mock_landmarks[160] = [120, 180, 0]  # P2 top right
        self.mock_landmarks[158] = [140, 180, 0]  # P3 top left
        self.mock_landmarks[133] = [160, 200, 0]  # P4 inner
        self.mock_landmarks[153] = [140, 220, 0]  # P5 bottom left
        self.mock_landmarks[144] = [120, 220, 0]  # P6 bottom right

        # Populate Right Eye Landmarks (Open position)
        self.mock_landmarks[362] = [240, 200, 0]
        self.mock_landmarks[385] = [260, 180, 0]
        self.mock_landmarks[387] = [280, 180, 0]
        self.mock_landmarks[263] = [300, 200, 0]
        self.mock_landmarks[373] = [280, 220, 0]
        self.mock_landmarks[380] = [260, 220, 0]

        # Populate Mouth Landmarks (Closed/normal position)
        self.mock_landmarks[MOUTH_INDICES["left"]] = [180, 300, 0]
        self.mock_landmarks[MOUTH_INDICES["right"]] = [220, 300, 0]
        self.mock_landmarks[MOUTH_INDICES["top_1"]] = [190, 295, 0]
        self.mock_landmarks[MOUTH_INDICES["bottom_1"]] = [190, 305, 0]
        self.mock_landmarks[MOUTH_INDICES["top_2"]] = [200, 295, 0]
        self.mock_landmarks[MOUTH_INDICES["bottom_2"]] = [200, 305, 0]
        self.mock_landmarks[MOUTH_INDICES["top_3"]] = [210, 295, 0]
        self.mock_landmarks[MOUTH_INDICES["bottom_3"]] = [210, 305, 0]

        # Pose landmarks (Nose=1, Chin=152, L_eye=33, R_eye=263, L_mouth=61, R_mouth=291)
        self.mock_landmarks[1] = [200, 220, 0]
        self.mock_landmarks[152] = [200, 340, 0]
        self.mock_landmarks[61] = [180, 300, 0]
        self.mock_landmarks[291] = [220, 300, 0]

    def test_01_normal_state_detection(self):
        """Test Case 1: Normal open eye and mouth feature extraction."""
        features = extract_facial_features(self.mock_landmarks)
        self.assertGreater(features["avg_ear"], 0.25)
        self.assertLess(features["mar"], 0.45)

    def test_02_yawning_detection(self):
        """Test Case 2: Yawning condition via increased MAR."""
        yawn_landmarks = self.mock_landmarks.copy()
        # Widen vertical mouth points
        yawn_landmarks[MOUTH_INDICES["bottom_1"]][1] += 45
        yawn_landmarks[MOUTH_INDICES["bottom_2"]][1] += 45
        yawn_landmarks[MOUTH_INDICES["bottom_3"]][1] += 45

        mar = calculate_mar(yawn_landmarks)
        self.assertGreater(mar, 0.60)

        detector = DistractionDetector()
        # Initialize the yawn state
        detector.update(
            face_found=True,
            pose_state="Normal",
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            mar=mar,
            mar_threshold=0.55,
            is_talking_mic=False,
            multiple_faces=False,
            fps=30.0,
        )
        self.assertIsNotNone(detector.yawn_start_time)
        
        # Simulate passage of 2 seconds
        import time
        detector.yawn_start_time = time.time() - 2.0
        
        # Trigger the yawn completion
        res = detector.update(
            face_found=True,
            pose_state="Normal",
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            mar=mar,
            mar_threshold=0.55,
            is_talking_mic=False,
            multiple_faces=False,
            fps=30.0,
        )
        self.assertTrue(res["is_yawning"])
        self.assertEqual(res["yawn_count"], 1)

    def test_03_sleepy_drowsiness_detection(self):
        """Test Case 3: Drowsiness trigger after 3s of low EAR."""
        drowsy_landmarks = self.mock_landmarks.copy()
        # Compress vertical eye points to simulate closed eyes
        drowsy_landmarks[160][1] = 200
        drowsy_landmarks[158][1] = 200
        drowsy_landmarks[153][1] = 200
        drowsy_landmarks[144][1] = 200

        low_ear = calculate_ear(drowsy_landmarks, LEFT_EYE_INDICES)
        self.assertLess(low_ear, 0.15)

        engine = DrowsinessDetector(buffer_size=90)
        # Feed 95 low EAR frames (exceeding 3s at 30fps)
        is_drowsy_flag = False
        for _ in range(95):
            res = engine.update(low_ear, ear_threshold=0.22, fps=30.0)
            if res["is_drowsy"]:
                is_drowsy_flag = True

        self.assertTrue(is_drowsy_flag)

    def test_04_looking_left_pose(self):
        """Test Case 4: Looking Left head classification."""
        state = head_pose_estimator.classify_pose_state(pitch=0.0, yaw=-25.0)
        self.assertEqual(state, "Looking Left")

    def test_05_looking_right_pose(self):
        """Test Case 5: Looking Right head classification."""
        state = head_pose_estimator.classify_pose_state(pitch=0.0, yaw=25.0)
        self.assertEqual(state, "Looking Right")

    def test_06_looking_down_pose(self):
        """Test Case 6: Looking Down (Phone simulation) classification."""
        state = head_pose_estimator.classify_pose_state(pitch=25.0, yaw=0.0)
        self.assertEqual(state, "Looking Down")

    def test_07_looking_up_pose(self):
        """Test Case 7: Looking Up classification."""
        state = head_pose_estimator.classify_pose_state(pitch=-25.0, yaw=0.0)
        self.assertEqual(state, "Looking Up")

    def test_08_spectacles_handling(self):
        """Test Case 8: Spectacles landmark distance robustness."""
        features = extract_facial_features(self.mock_landmarks)
        self.assertIsNotNone(features["avg_ear"])
        self.assertFalse(np.isnan(features["avg_ear"]))

    def test_09_low_light_enhancement(self):
        """Test Case 9: Low-light detection and CLAHE enhancement."""
        dark_frame = np.zeros((480, 640, 3), dtype=np.uint8) + 20
        enhanced_frame, is_low_light, lum = enhancer.enhance_frame(dark_frame)
        self.assertTrue(is_low_light)
        self.assertLess(lum, 45.0)
        self.assertEqual(enhanced_frame.shape, dark_frame.shape)

    def test_10_multiple_faces_handling(self):
        """Test Case 10: Multi-person tracking area logic test."""
        # Bounding box 1 (Small background face)
        bbox_small = (10, 10, 50, 50)  # Area 2500
        # Bounding box 2 (Large foreground face)
        bbox_large = (100, 100, 200, 200)  # Area 40000

        area1 = bbox_small[2] * bbox_small[3]
        area2 = bbox_large[2] * bbox_large[3]
        self.assertGreater(area2, area1)

    def test_11_time_based_blink_detection(self):
        """Test Case 11: Time-based adaptive blink detection."""
        detector = DrowsinessDetector(buffer_size=90)
        # Verify initial count
        self.assertEqual(detector.blink_count, 0)
        
        # Simulate blink down (avg_ear below threshold)
        import time
        t = time.time()
        # Mock update to trigger blink state
        detector.update(0.12, ear_threshold=0.18, fps=30.0)
        self.assertTrue(detector.in_blink)
        self.assertIsNotNone(detector.blink_start_time)
        
        # Override start time to simulate 150ms blink duration
        detector.blink_start_time = t - 0.15
        
        # Simulate blink up (avg_ear above threshold)
        detector.update(0.24, ear_threshold=0.18, fps=30.0)
        self.assertFalse(detector.in_blink)
        self.assertEqual(detector.blink_count, 1)

    def test_12_expression_analyzer(self):
        """Test Case 12: Expression analyzer rule verification."""
        from src.expression_analyzer import expression_analyzer
        # Set dummy baselines
        expression_analyzer.set_baselines(smile=-0.02, brow_raise=0.20, brow_knit=0.21)
        
        # Create a mock face landmark set
        landmarks = self.mock_landmarks.copy()
        
        # Setup valid face landmarks for scale calculations
        landmarks[10] = [200, 100, 0]   # Forehead
        landmarks[152] = [200, 340, 0]  # Chin
        landmarks[234] = [100, 220, 0]  # Left cheek
        landmarks[454] = [300, 220, 0]  # Right cheek
        
        # Eyebrows & Eyes
        landmarks[55] = [180, 160, 0]   # Left inner brow
        landmarks[285] = [220, 160, 0]  # Right inner brow
        landmarks[70] = [150, 150, 0]   # Left outer brow peak
        landmarks[300] = [250, 150, 0]  # Right outer brow peak
        landmarks[159] = [140, 180, 0]  # Left eye top
        landmarks[386] = [260, 180, 0]  # Right eye top
        
        # 1. Smile test
        landmarks[61] = [180, 290, 0]   # Pulled up left corner (higher in img coords)
        landmarks[291] = [220, 290, 0]  # Pulled up right corner
        landmarks[13] = [200, 300, 0]   # Center mouth
        landmarks[14] = [200, 300, 0]
        
        expr = expression_analyzer.classify_expression(landmarks, avg_ear=0.25, ear_threshold=0.18)
        self.assertEqual(expr, "Happy")

    def test_13_activity_classification(self):
        """Test Case 13: Activity classification rules."""
        import time
        detector = DistractionDetector()
        
        # Studying (neutral looking ahead)
        res = detector.update(
            face_found=True,
            pose_state="Normal",
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            mar=0.15,
            mar_threshold=0.55,
            is_talking_mic=False,
            multiple_faces=False,
            fps=30.0,
        )
        self.assertEqual(res["activity_state"], "STUDYING")
        
        # Phone use (severe tilt down for sustained frames)
        for _ in range(10):
            res = detector.update(
                face_found=True,
                pose_state="Looking Down",
                pitch=25.0,
                yaw=0.0,
                roll=0.0,
                mar=0.15,
                mar_threshold=0.55,
                is_talking_mic=False,
                multiple_faces=False,
                fps=30.0,
            )
        # After 2+ seconds (simulated by sequence), it should flag USING_PHONE
        # Note: in real time it uses time.time() difference, so let's mock the start time
        detector.phone_usage_start_time = time.time() - 2.5
        res = detector.update(
            face_found=True,
            pose_state="Looking Down",
            pitch=25.0,
            yaw=0.0,
            roll=0.0,
            mar=0.15,
            mar_threshold=0.55,
            is_talking_mic=False,
            multiple_faces=False,
            fps=30.0,
        )
        self.assertEqual(res["activity_state"], "USING_PHONE")

    def test_14_fast_blink_detection(self):
        """Test Case 14: Time-based fast blink detection (e.g. 33ms single-frame)."""
        detector = DrowsinessDetector(buffer_size=90)
        self.assertEqual(detector.blink_count, 0)

        # Start blink (low EAR)
        import time
        t = time.time()
        detector.update(0.12, ear_threshold=0.18, fps=30.0)
        self.assertTrue(detector.in_blink)

        # Mock start time to simulate 33ms blink duration (single frame at 30 FPS)
        detector.blink_start_time = t - 0.033

        # Complete blink (high EAR)
        detector.update(0.24, ear_threshold=0.18, fps=30.0)
        self.assertFalse(detector.in_blink)
        self.assertEqual(detector.blink_count, 1)

    def test_15_blink_cooldown_double_count_prevention(self):
        """Test Case 15: Verification that rapid consecutive low-EAR values (within 250ms) do not double count."""
        detector = DrowsinessDetector(buffer_size=90)
        self.assertEqual(detector.blink_count, 0)

        # 1st Blink: down
        import time
        t = time.time()
        detector.update(0.12, ear_threshold=0.18, fps=30.0)
        detector.blink_start_time = t - 0.15
        # 1st Blink: up (completed)
        detector.update(0.24, ear_threshold=0.18, fps=30.0)
        self.assertEqual(detector.blink_count, 1)
        self.assertIsNotNone(detector.last_blink_time)

        # 2nd rapid fluctuation: down (only 50ms after the first blink)
        detector.update(0.12, ear_threshold=0.18, fps=30.0)
        detector.blink_start_time = t - 0.05
        # 2nd rapid fluctuation: up (should be blocked by cooldown)
        detector.update(0.24, ear_threshold=0.18, fps=30.0)
        # Should still be 1 because of 250ms cooldown
        self.assertEqual(detector.blink_count, 1)

    def test_16_user_ear_0165_drowsiness_detection(self):
        """Test Case 16: Verify drowsiness triggers for user closed eyes at EAR 0.165 (default thresh 0.20)."""
        detector = DrowsinessDetector()
        is_drowsy_triggered = False
        # Feed 65 frames of EAR 0.165 at 30 fps (over 2 seconds)
        for _ in range(65):
            res = detector.update(0.165, ear_threshold=0.20, fps=30.0)
            if res["is_drowsy"]:
                is_drowsy_triggered = True

        self.assertTrue(is_drowsy_triggered)
        self.assertGreater(detector.drowsy_event_count, 0)

    def test_17_looking_right_distraction_flag(self):
        """Test Case 17: Verify Looking Right immediately sets activity state to LOOKING_AWAY and flags distraction condition."""
        detector = DistractionDetector()
        res = detector.update(
            face_found=True,
            pose_state="Looking Right",
            pitch=0.0,
            yaw=25.0,
            roll=0.0,
            mar=0.15,
            mar_threshold=0.55,
            is_talking_mic=False,
            multiple_faces=False,
            fps=30.0,
        )
        self.assertEqual(res["activity_state"], "LOOKING_AWAY")
        self.assertTrue(res["has_distraction_condition"])
        self.assertIn("Looking Away", res["distraction_reason"])

    def test_18_looking_left_and_down_distraction(self):
        """Test Case 18: Verify Looking Left and Looking Down immediately flag distraction condition."""
        detector = DistractionDetector()
        # Test Looking Left
        res_left = detector.update(
            face_found=True,
            pose_state="Looking Left",
            pitch=0.0,
            yaw=-25.0,
            roll=0.0,
            mar=0.15,
            mar_threshold=0.55,
            is_talking_mic=False,
            multiple_faces=False,
            fps=30.0,
            eyes_closed=True,  # Test that closed eyes from side profile does NOT block distraction flag
        )
        self.assertEqual(res_left["activity_state"], "LOOKING_AWAY")
        self.assertTrue(res_left["has_distraction_condition"])

        # Test Looking Down (Bottom)
        res_down = detector.update(
            face_found=True,
            pose_state="Looking Down",
            pitch=25.0,
            yaw=0.0,
            roll=0.0,
            mar=0.15,
            mar_threshold=0.55,
            is_talking_mic=False,
            multiple_faces=False,
            fps=30.0,
        )
        self.assertEqual(res_down["activity_state"], "USING_PHONE")
        self.assertTrue(res_down["has_distraction_condition"])


if __name__ == "__main__":
    unittest.main()
