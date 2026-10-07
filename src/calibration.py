"""
Adaptive Calibration Module.
Collects baseline facial feature metrics over an initial period (e.g. 10s)
to automatically compute personalized thresholds for EAR, MAR, and Head Pose.
"""
import time
import numpy as np
from src.config import SETTINGS
from utils.logger import logger


class AdaptiveCalibrator:
    """Manages baseline metric collection and dynamic threshold personalization."""

    def __init__(self, calibration_duration_sec: float = 10.0):
        self.duration_sec = calibration_duration_sec
        self.is_calibrating = False
        self.is_calibrated = False
        self.start_time = None

        self.ear_samples = []
        self.mar_samples = []
        self.pitch_samples = []
        self.yaw_samples = []
        self.smile_samples = []
        self.brow_raise_samples = []
        self.brow_knit_samples = []

        self.calibrated_thresholds = {
            "ear_threshold": SETTINGS["thresholds"]["default_ear_threshold"],
            "mar_threshold": SETTINGS["thresholds"]["default_mar_threshold"],
            "pitch_offset": 11.4,
            "yaw_offset": 0.0,
            "smile_baseline": -0.02,
            "brow_raise_baseline": 0.20,
            "brow_knit_baseline": 0.21,
        }
        self.auto_calibrated_thresholds = dict(self.calibrated_thresholds)
        self.is_manual_override = False
        self.just_finished = False


    def start_calibration(self):
        """Reset samples and start new calibration session."""
        self.ear_samples = []
        self.mar_samples = []
        self.pitch_samples = []
        self.yaw_samples = []
        self.smile_samples = []
        self.brow_raise_samples = []
        self.brow_knit_samples = []
        self.start_time = time.time()
        self.is_calibrating = True
        self.is_calibrated = False
        self.just_finished = False
        logger.info("Adaptive Calibration started.")


    def update(
        self,
        avg_ear: float,
        mar: float,
        pitch: float,
        yaw: float,
        landmarks: np.ndarray = None,
    ) -> float:
        """
        Process a new frame during calibration.

        Returns:
            progress (float): Progress percentage (0.0 to 100.0)
        """
        if not self.is_calibrating:
            return 100.0 if self.is_calibrated else 0.0

        elapsed = time.time() - self.start_time
        progress = min(100.0, (elapsed / self.duration_sec) * 100.0)

        # Collect valid samples
        if avg_ear > 0.10:  # Ignore fully closed blinks during calibration baseline
            self.ear_samples.append(avg_ear)
        if mar > 0.0:
            self.mar_samples.append(mar)

        self.pitch_samples.append(pitch)
        self.yaw_samples.append(yaw)

        # Collect expression samples
        if landmarks is not None:
            from src.expression_analyzer import expression_analyzer
            metrics = expression_analyzer.extract_expression_metrics(landmarks)
            if metrics["smile_score"] != 0.0 or metrics["brow_raise_score"] != 0.0:
                self.smile_samples.append(metrics["smile_score"])
                self.brow_raise_samples.append(metrics["brow_raise_score"])
                self.brow_knit_samples.append(metrics["brow_knit_score"])

        if elapsed >= self.duration_sec:
            self.finalize_calibration()

        return progress

    def finalize_calibration(self):
        """Compute personalized thresholds from collected samples."""
        self.is_calibrating = False
        self.is_calibrated = True
        self.just_finished = True


        if self.ear_samples:
            mean_ear = np.mean(self.ear_samples)
            # Threshold set to 85% of open EAR baseline (clamped between 0.18 and 0.25)
            computed_ear_thresh = max(0.18, min(0.25, mean_ear * 0.85))
            self.calibrated_thresholds["ear_threshold"] = round(float(computed_ear_thresh), 4)
        else:
            mean_ear = SETTINGS["thresholds"]["default_ear_threshold"]

        if self.mar_samples:
            mean_mar = np.mean(self.mar_samples)
            # Threshold set to baseline mouth plus opening margin
            computed_mar_thresh = max(0.45, min(0.70, mean_mar + 0.28))
            self.calibrated_thresholds["mar_threshold"] = round(float(computed_mar_thresh), 4)

        if self.pitch_samples:
            self.calibrated_thresholds["pitch_offset"] = round(float(np.mean(self.pitch_samples)), 2)

        if self.yaw_samples:
            self.calibrated_thresholds["yaw_offset"] = round(float(np.mean(self.yaw_samples)), 2)

        # Finalize expression baselines
        if self.smile_samples:
            self.calibrated_thresholds["smile_baseline"] = round(float(np.mean(self.smile_samples)), 4)
        if self.brow_raise_samples:
            self.calibrated_thresholds["brow_raise_baseline"] = round(float(np.mean(self.brow_raise_samples)), 4)
        if self.brow_knit_samples:
            self.calibrated_thresholds["brow_knit_baseline"] = round(float(np.mean(self.brow_knit_samples)), 4)

        # Propagate to expression analyzer
        from src.expression_analyzer import expression_analyzer
        expression_analyzer.set_baselines(
            self.calibrated_thresholds["smile_baseline"],
            self.calibrated_thresholds["brow_raise_baseline"],
            self.calibrated_thresholds["brow_knit_baseline"],
        )

        # Save baseline copy for deactivation / reset functionality
        self.auto_calibrated_thresholds = dict(self.calibrated_thresholds)
        self.is_manual_override = False

        logger.info(
            f"Adaptive Calibration finished. Calibrated Thresholds: {self.calibrated_thresholds}"
        )

    def get_thresholds(self) -> dict:
        """Return active calibrated thresholds."""
        return self.calibrated_thresholds

    def set_manual_thresholds(self, ear_threshold: float, mar_threshold: float):
        """Manually set active EAR and MAR thresholds from UI sliders."""
        self.calibrated_thresholds["ear_threshold"] = round(float(ear_threshold), 4)
        self.calibrated_thresholds["mar_threshold"] = round(float(mar_threshold), 4)
        self.is_manual_override = True

    def deactivate_manual_override(self):
        """Restore thresholds to the last auto-calibrated baseline (or default settings)."""
        self.calibrated_thresholds["ear_threshold"] = self.auto_calibrated_thresholds["ear_threshold"]
        self.calibrated_thresholds["mar_threshold"] = self.auto_calibrated_thresholds["mar_threshold"]
        self.is_manual_override = False
        logger.info("Manual Admin Override deactivated. Restored auto-calibrated baseline.")


