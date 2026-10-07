"""
Distraction & Yawning Detection Engine.
Tracks head orientation, face presence, phone usage posture,
and sustained mouth aspect ratio (MAR) to detect distraction and yawns.
"""
import time
from src.config import SETTINGS
from utils.logger import logger


from collections import deque
import numpy as np
import time
from src.config import SETTINGS
from utils.logger import logger


class DistractionDetector:
    """Detects head distraction, missing face distraction, phone simulation, yawning, and detailed student activities."""

    def __init__(self):
        # Distraction tracking
        self.is_distracted = False
        self.distraction_start_time = None
        self.total_distraction_duration = 0.0
        self.distraction_count = 0
        self.distraction_reason = ""
        self.distracted_frames = 0
        self.distraction_clear_frames = 0
        self.longest_distraction = 0.0

        # Yawning tracking
        self.is_yawning = False
        self.yawn_start_time = None
        self.yawn_closed_start_time = None
        self.yawn_count = 0
        self.smoothed_mar = 0.15

        # Activity classification and history tracking
        self.activity_state = "STUDYING"
        self.pose_history = deque(maxlen=60)  # 2 seconds at 30 fps
        self.mar_history = deque(maxlen=30)   # 1 second at 30 fps
        self.phone_usage_start_time = None
        self.look_away_start_time = None
        self.no_face_start_time = None

    def update(
        self,
        face_found: bool,
        pose_state: str,
        pitch: float,
        yaw: float,
        roll: float,
        mar: float,
        mar_threshold: float,
        is_talking_mic: bool = False,
        multiple_faces: bool = False,
        fps: float = 30.0,
        eyes_closed: bool = False,
        is_drowsy: bool = False,
    ) -> dict:
        """
        Update state with current frame inputs.

        Returns:
            dict containing:
                - is_distracted (bool)
                - distraction_reason (str)
                - distraction_count (int)
                - total_distraction_duration (float)
                - is_yawning (bool)
                - yawn_count (int)
                - activity_state (str)
        """
        curr_time = time.time()
        distraction_sec = SETTINGS["temporal"]["distraction_duration_sec"]
        distraction_frames_thresh = int(distraction_sec * fps)

        yawn_sec = SETTINGS["temporal"]["yawn_duration_sec"]

        # 1. Activity Classification
        if not face_found:
            self.pose_history.clear()
            self.mar_history.clear()
            self.phone_usage_start_time = None
            self.look_away_start_time = None
            
            if self.no_face_start_time is None:
                self.no_face_start_time = curr_time
            
            if curr_time - self.no_face_start_time >= 2.0:
                self.activity_state = "NO_FACE"
            else:
                self.activity_state = "STUDYING"  # brief loss is ignored
        else:
            self.no_face_start_time = None
            self.pose_history.append((pitch, yaw, roll))
            self.mar_history.append(mar)

            # Calculate head pose movement variance
            if len(self.pose_history) >= 15:
                pitch_std = np.std([p[0] for p in self.pose_history])
                yaw_std = np.std([p[1] for p in self.pose_history])
                roll_std = np.std([p[2] for p in self.pose_history])
                pose_movement = pitch_std + yaw_std + roll_std
            else:
                pose_movement = 0.0

            # Determine talking from mar variance (mouth movement)
            is_talking_mouth = False
            if len(self.mar_history) >= 20:
                mar_std = np.std(self.mar_history)
                mar_mean = np.mean(self.mar_history)
                # Normal speaking causes mouth to open and close (high std)
                # but not as wide or sustained as a yawn
                if mar_std > 0.025 and 0.14 <= mar_mean <= 0.40 and mar < mar_threshold:
                    is_talking_mouth = True

            # Head pose threshold settings
            yaw_left = SETTINGS["thresholds"]["yaw_left_threshold"]
            yaw_right = SETTINGS["thresholds"]["yaw_right_threshold"]
            pitch_down = SETTINGS["thresholds"].get("pitch_down_threshold", 18.0)
            pitch_up = SETTINGS["thresholds"].get("pitch_up_threshold", -18.0)

            # Looking away with 3-degree hysteresis (looking up, left, right)
            if self.activity_state == "LOOKING_AWAY":
                is_looking_away = (yaw < (yaw_left + 3.0)) or (yaw > (yaw_right - 3.0)) or (pitch < (pitch_up + 3.0))
            else:
                is_looking_away = (yaw < yaw_left) or (yaw > yaw_right) or (pitch < pitch_up)

            # Phone usage (pitch down) with 4-degree hysteresis
            if self.activity_state == "USING_PHONE":
                is_pitch_down = pitch > (pitch_down - 4.0)
            else:
                is_pitch_down = pitch > pitch_down

            # State Hierarchy Classification
            if is_drowsy:
                self.phone_usage_start_time = None
                self.look_away_start_time = None
                self.activity_state = "STUDYING"
            elif multiple_faces:
                self.activity_state = "MULTIPLE_FACES"
                self.phone_usage_start_time = None
                self.look_away_start_time = None
            elif is_looking_away:
                self.phone_usage_start_time = None
                self.look_away_start_time = None
                self.activity_state = "LOOKING_AWAY"
            elif is_pitch_down:
                self.look_away_start_time = None
                self.phone_usage_start_time = None
                self.activity_state = "USING_PHONE"
            elif is_talking_mic or is_talking_mouth:
                self.phone_usage_start_time = None
                self.look_away_start_time = None
                self.activity_state = "TALKING"
            elif pose_movement > 9.0:
                self.phone_usage_start_time = None
                self.look_away_start_time = None
                self.activity_state = "PLAYING_FIDGETING"
            elif 6.0 <= pitch <= pitch_down and abs(yaw) <= 12.0:
                self.phone_usage_start_time = None
                self.look_away_start_time = None
                self.activity_state = "WRITING_READING"
            else:
                self.phone_usage_start_time = None
                self.look_away_start_time = None
                self.activity_state = "STUDYING"

        # 2. Distraction Condition Evaluation
        has_distraction_condition = False
        reason = ""

        if self.activity_state == "NO_FACE":
            has_distraction_condition = True
            reason = "Face Not Found"
        elif self.activity_state == "MULTIPLE_FACES":
            has_distraction_condition = True
            reason = "Multiple Faces Detected (Collusion)"
        elif self.activity_state == "USING_PHONE":
            has_distraction_condition = True
            reason = "Mobile Phone Usage"
        elif self.activity_state == "LOOKING_AWAY":
            has_distraction_condition = True
            reason = f"Looking Away ({pose_state})"
        elif self.activity_state == "PLAYING_FIDGETING":
            has_distraction_condition = True
            reason = "Playing / Fidgeting"
        elif self.activity_state == "TALKING":
            has_distraction_condition = True
            reason = "Talking / Speech Detected"

        distraction_event_triggered = False
        if has_distraction_condition:
            self.distraction_clear_frames = 0
            self.distracted_frames += 1
            if self.distracted_frames >= distraction_frames_thresh:
                if not self.is_distracted:
                    self.is_distracted = True
                    self.distraction_start_time = curr_time
                    self.distraction_count += 1
                    distraction_event_triggered = True
                    logger.info(f"DISTRACTION DETECTED: {reason}")
                self.distraction_reason = reason
        else:
            if self.is_distracted:
                self.distraction_clear_frames += 1
                clear_thresh = max(15, int(1.5 * fps))
                if self.distraction_clear_frames >= clear_thresh:
                    self.is_distracted = False
                    if self.distraction_start_time:
                        dur = curr_time - self.distraction_start_time - (clear_thresh / fps)
                        dur = max(0.0, dur)
                        self.total_distraction_duration += dur
                        if dur > self.longest_distraction:
                            self.longest_distraction = dur
                        self.distraction_start_time = None
                    self.distracted_frames = 0
                    self.distraction_reason = ""
                    self.distraction_clear_frames = 0
            else:
                self.distracted_frames = 0
                self.distraction_reason = ""
                self.distraction_clear_frames = 0

        current_distraction_dur = 0.0
        if self.is_distracted and self.distraction_start_time:
            current_distraction_dur = curr_time - self.distraction_start_time
            if current_distraction_dur > self.longest_distraction:
                self.longest_distraction = current_distraction_dur

        # 3. Yawning Condition Evaluation (with Hysteresis & Time-based tracking)
        self.smoothed_mar = 0.25 * mar + 0.75 * self.smoothed_mar
        is_mar_high = self.smoothed_mar >= mar_threshold
        yawn_event_triggered = False

        if is_mar_high and face_found:
            self.yawn_closed_start_time = None
            if self.yawn_start_time is None:
                self.yawn_start_time = curr_time
            else:
                yawn_dur = curr_time - self.yawn_start_time
                if yawn_dur >= yawn_sec:
                    if not self.is_yawning:
                        self.is_yawning = True
                        self.yawn_count += 1
                        yawn_event_triggered = True
                        logger.info("YAWN DETECTED!")
        else:
            if self.is_yawning:
                if self.yawn_closed_start_time is None:
                    self.yawn_closed_start_time = curr_time
                yawn_closed_dur = curr_time - self.yawn_closed_start_time
                if yawn_closed_dur >= 1.0 or self.smoothed_mar < (mar_threshold - 0.15):
                    self.is_yawning = False
                    self.yawn_start_time = None
                    self.yawn_closed_start_time = None
            else:
                self.yawn_start_time = None

        return {
            "is_distracted": self.is_distracted,
            "has_distraction_condition": has_distraction_condition,
            "distraction_reason": reason if has_distraction_condition else self.distraction_reason,
            "distraction_count": self.distraction_count,
            "distraction_event_triggered": distraction_event_triggered,
            "total_distraction_duration": round(
                float(self.total_distraction_duration + current_distraction_dur), 2
            ),
            "longest_distraction": round(float(self.longest_distraction), 2),
            "is_yawning": self.is_yawning,
            "yawn_count": self.yawn_count,
            "yawn_event_triggered": yawn_event_triggered,
            "activity_state": self.activity_state,
        }

    def reset(self):
        """Reset state tracking."""
        self.is_distracted = False
        self.distraction_start_time = None
        self.total_distraction_duration = 0.0
        self.distraction_count = 0
        self.distraction_reason = ""
        self.distracted_frames = 0
        self.distraction_clear_frames = 0
        self.longest_distraction = 0.0
        self.is_yawning = False
        self.yawn_start_time = None
        self.yawn_count = 0
        self.yawn_frames = 0
        self.yawn_reset_cooldown_frames = 0
        self.smoothed_mar = 0.15
        self.activity_state = "STUDYING"
        self.pose_history.clear()
        self.mar_history.clear()
        self.phone_usage_start_time = None
        self.look_away_start_time = None
        self.no_face_start_time = None
