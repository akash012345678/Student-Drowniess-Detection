"""
Streamlit Dashboard Module.
Orchestrates live video feed processing, OpenCV annotations,
metric charts, sidebar controls, CSV exports, and session analytics.
"""
import datetime
import os
import tempfile
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.config import COLOR_CYAN, COLOR_GREEN, COLOR_RED, COLOR_WHITE, COLOR_YELLOW, SETTINGS
from src.image_processing import enhancer
from src.face_mesh import FaceMeshDetector
from src.feature_extraction import extract_facial_features
from src.head_pose import head_pose_estimator
from src.calibration import AdaptiveCalibrator
from src.drowsiness_detector import DrowsinessDetector
from src.distraction_detector import DistractionDetector
from src.attention_scorer import attention_scorer
from src.analytics import SessionAnalytics
from utils.exporter import SessionExporter
from utils.audio import trigger_audio_alert
from ui.components import inject_custom_css, render_header, render_status_pill
from utils.mic_detector import MicDetector
from src.expression_analyzer import expression_analyzer


def draw_visual_annotations(
    frame: np.ndarray,
    mesh_data: dict,
    features: dict,
    pose_data: dict,
    state: str,
    is_low_light: bool,
    is_calibrating: bool,
    calib_progress: float,
) -> np.ndarray:
    """Draw clean visual overlays (landmarks, bounding box, status pill, pose axis) on OpenCV frame."""
    if frame is None:
        return frame

    annotated = frame.copy()
    h, w, c = annotated.shape

    # 1. Low Light Indicator Overlay
    if is_low_light:
        cv2.putText(
            annotated,
            "LOW LIGHT ENHANCED (CLAHE)",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_YELLOW,
            1,
            cv2.LINE_AA,
        )

    # 2. Calibration Progress Overlay
    if is_calibrating:
        bar_w = int((w - 80) * (calib_progress / 100.0))
        cv2.rectangle(annotated, (40, 40), (w - 40, 65), (50, 50, 50), -1)
        cv2.rectangle(annotated, (40, 40), (40 + bar_w, 65), COLOR_CYAN, -1)
        cv2.putText(
            annotated,
            f"CALIBRATING BASELINE ({int(calib_progress)}%)",
            (50, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

    # 3. Multi-Face Warning Banner
    if mesh_data.get("multiple_faces"):
        cv2.rectangle(annotated, (20, 80), (w - 20, 115), (0, 0, 180), -1)
        cv2.putText(
            annotated,
            "WARNING: MULTIPLE FACES DETECTED - TRACKING PRIMARY FACE",
            (30, 103),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_WHITE,
            2,
            cv2.LINE_AA,
        )

    # 4. Face Bounding Box & Landmark Mesh Highlights
    if mesh_data.get("face_found") and mesh_data.get("landmarks_pixel") is not None:
        bbox = mesh_data["bbox"]
        if bbox:
            bx, by, bw, bh = bbox
            box_color = (
                COLOR_RED
                if state == "DROWSY"
                else (COLOR_YELLOW if state == "DISTRACTED" else COLOR_GREEN)
            )
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), box_color, 2)

        landmarks = mesh_data["landmarks_pixel"]

        # Draw Eye Contours
        left_eye_pts = landmarks[[33, 160, 158, 133, 153, 144], :2].astype(np.int32)
        right_eye_pts = landmarks[[362, 385, 387, 263, 373, 380], :2].astype(np.int32)
        cv2.polylines(annotated, [left_eye_pts], True, COLOR_CYAN, 1, cv2.LINE_AA)
        cv2.polylines(annotated, [right_eye_pts], True, COLOR_CYAN, 1, cv2.LINE_AA)

        # Draw Mouth Contour
        mouth_pts = landmarks[[61, 81, 13, 311, 291, 402, 14, 178], :2].astype(np.int32)
        cv2.polylines(annotated, [mouth_pts], True, COLOR_YELLOW, 1, cv2.LINE_AA)

        # Draw Pose Nose Direction Vector
        if pose_data.get("rotation_vector") is not None:
            rvec = pose_data["rotation_vector"]
            tvec = pose_data["translation_vector"]

            # 3D Nose Axis endpoints
            axis_3d = np.float32([[0, 0, 0], [0, 0, 150], [0, 150, 0], [150, 0, 0]])
            focal_length = w
            center = (w / 2.0, h / 2.0)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                dtype=np.float64,
            )
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
            nose_tip = tuple(imgpts[0].ravel().astype(int))
            z_axis = tuple(imgpts[1].ravel().astype(int))

            cv2.arrowedLine(annotated, nose_tip, z_axis, COLOR_CYAN, 2, cv2.LINE_AA, tipLength=0.2)

    return annotated


def run_dashboard():
    """Main dashboard loop and Streamlit application setup."""
    st.set_page_config(
        page_title="AI Student Drowsiness & Attention System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()
    render_header()

    # Initialize Session State Objects
    if "detector" not in st.session_state:
        st.session_state.detector = FaceMeshDetector()
        st.session_state.calibrator = AdaptiveCalibrator()
        st.session_state.drowsy_engine = DrowsinessDetector()
        st.session_state.distraction_engine = DistractionDetector()
        st.session_state.analytics = SessionAnalytics()
        st.session_state.exporter = SessionExporter()
        st.session_state.mic_detector = MicDetector()
        st.session_state.event_log = []
        st.session_state.running = False

    calibrator = st.session_state.calibrator
    drowsy_engine = st.session_state.drowsy_engine
    distraction_engine = st.session_state.distraction_engine
    analytics = st.session_state.analytics
    exporter = st.session_state.exporter

    # Sidebar Controls
    st.sidebar.markdown("### 🎛️ Control Panel")

    input_mode = st.sidebar.radio(
        "Select Input Source",
        options=["Webcam Feed", "Upload Video File"],
        index=0,
    )

    webcam_index = 0
    uploaded_file = None

    if input_mode == "Webcam Feed":
        webcam_index = st.sidebar.number_input("Webcam Device Index", min_value=0, max_value=5, value=0)
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload MP4/AVI Video (Recorded / NTHU Benchmark)",
            type=["mp4", "avi", "mov"],
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Adaptive Calibration")
    if st.sidebar.button("Recalibrate User Baseline (10s)", use_container_width=True):
        calibrator.start_calibration()
        st.session_state.running = True
        st.session_state.mic_detector.start()

    calib_status_placeholder = st.sidebar.empty()

    current_thresholds = calibrator.get_thresholds()
    if calibrator.is_calibrating:
        calib_status_placeholder.warning("⏳ Calibration initiated! Look normally at camera.")
    elif calibrator.is_calibrated:
        calib_status_placeholder.markdown(
            f"""
            <div style="background: rgba(0, 255, 135, 0.12); border: 1px solid #00ff87; padding: 10px 12px; border-radius: 8px; margin-top: 5px; margin-bottom: 10px;">
                <div style="color: #00ff87; font-weight: 600; font-size: 13px;">✅ Calibrated Thresholds Active</div>
                <div style="font-size: 12px; color: #e0e0e0; margin-top: 5px; line-height: 1.5;">
                    • <strong>EAR Closed:</strong> <code>{current_thresholds['ear_threshold']:.3f}</code><br>
                    • <strong>MAR Yawn:</strong> <code>{current_thresholds['mar_threshold']:.3f}</code><br>
                    • <strong>Pitch Offset:</strong> <code>{current_thresholds.get('pitch_offset', 0.0):+.1f}°</code> | <strong>Yaw:</strong> <code>{current_thresholds.get('yaw_offset', 0.0):+.1f}°</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        calib_status_placeholder.caption("Click button to calibrate baseline to your facial structure.")

    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔒 Threshold Governance")
    admin_mode_toggle = st.sidebar.toggle(
        "Enable Proctor / Admin Mode",
        value=st.session_state.admin_authenticated,
        help="Allows manual threshold override for instructors and proctors.",
    )

    if not admin_mode_toggle:
        st.session_state.admin_authenticated = False

    # Preserve calibrated thresholds when active
    if "prev_ear_slider" not in st.session_state:
        st.session_state.prev_ear_slider = float(current_thresholds["ear_threshold"])
    if "prev_mar_slider" not in st.session_state:
        st.session_state.prev_mar_slider = float(current_thresholds["mar_threshold"])

    if admin_mode_toggle:
        if not st.session_state.admin_authenticated:
            pwd_input = st.sidebar.text_input(
                "🔑 Enter Admin Password",
                type="password",
                key="admin_password_input",
                help="Enter instructor/proctor passcode.",
            )
            if st.sidebar.button("🔓 Authenticate & Unlock", use_container_width=True):
                expected_pwd = SETTINGS.get("admin_password", "admin")
                if pwd_input == expected_pwd:
                    st.session_state.admin_authenticated = True
                    st.sidebar.success("✅ Admin Access Granted!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Incorrect Password!")

    if admin_mode_toggle and st.session_state.admin_authenticated:
        st.sidebar.caption("⚙️ Manual Proctor Controls Unlocked")

        col_adm1, col_adm2 = st.sidebar.columns(2)
        with col_adm1:
            if st.button("🔒 Lock Admin", use_container_width=True):
                st.session_state.admin_authenticated = False
                st.rerun()
        with col_adm2:
            if st.button("🔄 Reset to Calib", use_container_width=True, help="Deactivate manual override and revert to user's calibrated baseline"):
                calibrator.deactivate_manual_override()
                current_thresh = calibrator.get_thresholds()
                st.session_state.prev_ear_slider = float(current_thresh["ear_threshold"])
                st.session_state.prev_mar_slider = float(current_thresh["mar_threshold"])
                st.sidebar.success("Reverted to Calibrated Baseline!")
                st.rerun()

        # Explicit Checkbox / Toggle for Manual Override
        override_active = st.sidebar.checkbox(
            "Enable Manual Threshold Override",
            value=calibrator.is_manual_override,
            help="Check to manually override thresholds with sliders below. Uncheck to deactivate manual threshold and return to user's calibrated baseline.",
        )

        if override_active != calibrator.is_manual_override:
            if not override_active:
                calibrator.deactivate_manual_override()
                current_thresh = calibrator.get_thresholds()
                st.session_state.prev_ear_slider = float(current_thresh["ear_threshold"])
                st.session_state.prev_mar_slider = float(current_thresh["mar_threshold"])
                st.rerun()
            else:
                calibrator.is_manual_override = True

        if calibrator.is_manual_override:
            st.sidebar.warning("⚠️ Manual Admin Override Active")
        else:
            st.sidebar.info("✅ Calibration Mode Active (Manual Override Deactivated)")

        current_thresholds = calibrator.get_thresholds()

        ear_thresh = st.sidebar.slider(
            "EAR Closed Threshold",
            min_value=0.12,
            max_value=0.35,
            value=float(current_thresholds["ear_threshold"]),
            step=0.01,
            disabled=not calibrator.is_manual_override,
            help="Check 'Enable Manual Threshold Override' above to enable slider adjustment." if not calibrator.is_manual_override else "Adjust EAR threshold manually.",
        )
        mar_thresh = st.sidebar.slider(
            "MAR Yawn Threshold",
            min_value=0.35,
            max_value=0.85,
            value=float(current_thresholds["mar_threshold"]),
            step=0.02,
            disabled=not calibrator.is_manual_override,
            help="Check 'Enable Manual Threshold Override' above to enable slider adjustment." if not calibrator.is_manual_override else "Adjust MAR threshold manually.",
        )

        # Only update manual thresholds if override is active, calibrator is NOT calibrating, AND slider was changed
        if calibrator.is_manual_override and not calibrator.is_calibrating:
            if (
                ear_thresh != st.session_state.prev_ear_slider
                or mar_thresh != st.session_state.prev_mar_slider
            ):
                st.session_state.prev_ear_slider = ear_thresh
                st.session_state.prev_mar_slider = mar_thresh
                calibrator.set_manual_thresholds(ear_thresh, mar_thresh)

    if not st.session_state.admin_authenticated:
        st.sidebar.caption("🔒 Thresholds locked by Proctor policy. Calibration active.")
        ear_thresh = st.sidebar.slider(
            "EAR Closed Threshold (Locked)",
            min_value=0.12,
            max_value=0.35,
            value=float(current_thresholds["ear_threshold"]),
            step=0.01,
            disabled=True,
            help="Locked to prevent tampering. Enable Proctor Mode & enter password to adjust.",
        )
        mar_thresh = st.sidebar.slider(
            "MAR Yawn Threshold (Locked)",
            min_value=0.35,
            max_value=0.85,
            value=float(current_thresholds["mar_threshold"]),
            step=0.02,
            disabled=True,
            help="Locked to prevent tampering. Enable Proctor Mode & enter password to adjust.",
        )



    # Session Control Buttons
    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        if st.button("▶ Start Stream", use_container_width=True):
            st.session_state.running = True
            st.session_state.mic_detector.start()
    with col_btn2:
        if st.button("⏹ Stop Stream", use_container_width=True):
            st.session_state.running = False
            st.session_state.mic_detector.stop()

    # Main UI Layout Columns
    col_video, col_metrics = st.columns([1.8, 1.2])

    with col_video:
        status_placeholder = st.empty()
        video_placeholder = st.empty()
        chart_placeholder = st.empty()

    with col_metrics:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Real-Time Telemetry")

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            score_metric = st.empty()
            ear_metric = st.empty()
            blink_metric = st.empty()
            drowsy_dur_metric = st.empty()
        with m_col2:
            pose_metric = st.empty()
            mar_metric = st.empty()
            yawn_metric = st.empty()
            distraction_metric = st.empty()

        st.markdown("---")
        session_time_metric = st.empty()
        focus_pct_metric = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 📜 Recent Event Log")
        event_log_placeholder = st.empty()

    # Handle Video Capture Input
    cap = None
    if st.session_state.running:
        if input_mode == "Webcam Feed":
            cap = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(webcam_index, cv2.CAP_ANY)
            if cap is not None:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()
            cap = cv2.VideoCapture(tfile.name)

    prev_time = time.time()
    frame_counter = 0
    prev_state_logged = None

    # Video Loop Execution
    while st.session_state.running and cap is not None and cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            st.warning("End of video stream or camera disconnected.")
            st.session_state.running = False
            break

        frame_counter += 1
        curr_time = time.time()
        fps = 1.0 / max(1e-5, (curr_time - prev_time))
        prev_time = curr_time

        # 1. Low Light Preprocessing & Enhancement
        enhanced_frame, is_low_light, luminance = enhancer.enhance_frame(raw_frame)

        # 2. MediaPipe FaceMesh Landmark Processing
        mesh_data = st.session_state.detector.process_frame(enhanced_frame)

        # Default Values
        current_state = "ALERT"
        features = {"left_ear": 0.0, "right_ear": 0.0, "avg_ear": 0.0, "mar": 0.0}
        active_thresholds = calibrator.get_thresholds()
        active_ear_thresh = active_thresholds["ear_threshold"]
        active_mar_thresh = active_thresholds["mar_threshold"]
        pose_data = {
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "pose_state": "Normal",
            "rotation_vector": None,
            "translation_vector": None,
        }

        if mesh_data["face_found"]:
            # 3. Feature Extraction
            features = extract_facial_features(mesh_data["landmarks_pixel"])

            # 4. Head Pose Estimation
            pose_data = head_pose_estimator.estimate_pose(
                mesh_data["landmarks_pixel"], enhanced_frame.shape[:2]
            )

            # 5. Adaptive Calibration Step
            if calibrator.is_calibrating:
                calib_progress = calibrator.update(
                    features["avg_ear"],
                    features["mar"],
                    pose_data["pitch"],
                    pose_data["yaw"],
                    landmarks=mesh_data["landmarks_pixel"],
                )
                if frame_counter % 3 == 0:
                    calib_status_placeholder.warning(
                        f"⏳ Calibrating... ({int(calib_progress)}%)\nLook normally at camera."
                    )
            else:
                calib_progress = 100.0

            active_thresholds = calibrator.get_thresholds()
            active_ear_thresh = active_thresholds["ear_threshold"]
            active_mar_thresh = active_thresholds["mar_threshold"]
            pitch_offset = active_thresholds.get("pitch_offset", 0.0)
            yaw_offset = active_thresholds.get("yaw_offset", 0.0)

            if getattr(calibrator, "just_finished", False):
                calibrator.just_finished = False
                st.session_state.prev_ear_slider = float(active_ear_thresh)
                st.session_state.prev_mar_slider = float(active_mar_thresh)

                st.session_state.event_log.append(
                    {
                        "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "Event": "🎯 BASELINE CALIBRATED",
                        "Details": f"EAR: {active_ear_thresh:.3f}, MAR: {active_mar_thresh:.3f}",
                    }
                )

            if calibrator.is_calibrated and not calibrator.is_calibrating and (frame_counter % 15 == 0):
                calib_status_placeholder.markdown(
                    f"""
                    <div style="background: rgba(0, 255, 135, 0.12); border: 1px solid #00ff87; padding: 10px 12px; border-radius: 8px; margin-top: 5px; margin-bottom: 10px;">
                        <div style="color: #00ff87; font-weight: 600; font-size: 13px;">✅ Calibrated Thresholds Active</div>
                        <div style="font-size: 12px; color: #e0e0e0; margin-top: 5px; line-height: 1.5;">
                            • <strong>EAR Closed:</strong> <code>{active_ear_thresh:.3f}</code><br>
                            • <strong>MAR Yawn:</strong> <code>{active_mar_thresh:.3f}</code><br>
                            • <strong>Pitch Offset:</strong> <code>{pitch_offset:+.1f}°</code> | <strong>Yaw:</strong> <code>{yaw_offset:+.1f}°</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            pitch_adj = pose_data["pitch"] - pitch_offset
            yaw_adj = pose_data["yaw"] - yaw_offset
            
            # Re-classify pose state based on normalized head pose
            pose_state_adj = head_pose_estimator.classify_pose_state(pitch_adj, yaw_adj)

            # 6. Drowsiness Detector Engine
            drowsy_res = drowsy_engine.update(
                features["avg_ear"], active_ear_thresh, fps=fps
            )

            # 7. Distraction & Yawning Detector Engine
            is_talking = st.session_state.mic_detector.is_talking
            multiple_faces = mesh_data.get("multiple_faces", False)
            distract_res = distraction_engine.update(
                face_found=True,
                pose_state=pose_state_adj,
                pitch=pitch_adj,
                yaw=yaw_adj,
                roll=pose_data["roll"],
                mar=features["mar"],
                mar_threshold=active_mar_thresh,
                is_talking_mic=is_talking,
                multiple_faces=multiple_faces,
                fps=fps,
                eyes_closed=drowsy_res.get("eyes_closed", False),
                is_drowsy=drowsy_res["is_drowsy"],
            )

            # Facial Expression Classification
            expression = expression_analyzer.classify_expression(
                mesh_data["landmarks_pixel"], features["avg_ear"], active_ear_thresh
            )

            # Log Yawn Events independently to guarantee entry in event_log
            if distract_res.get("yawn_event_triggered"):
                st.session_state.event_log.append(
                    {
                        "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "Event": "🥱 YAWN DETECTED",
                        "Details": f"MAR: {features['mar']:.3f} (> {active_mar_thresh:.3f})",
                    }
                )

            # Determine State Priority (DROWSY > DISTRACTED > ALERT)
            if drowsy_res["is_drowsy"]:
                current_state = "DROWSY"
                if drowsy_res["event_triggered"]:
                    # Capture automatic screenshot & sound alert
                    shot_path = exporter.capture_drowsy_screenshot(enhanced_frame)
                    trigger_audio_alert("drowsy")
                    st.session_state.event_log.append(
                        {
                            "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                            "Event": "🚨 DROWSINESS DETECTED",
                            "Details": f"EAR: {features['avg_ear']:.3f} (< {active_ear_thresh:.3f})",
                        }
                    )
            elif distract_res["is_distracted"]:
                current_state = "DISTRACTED"
                if distract_res.get("distraction_event_triggered"):
                    trigger_audio_alert("distracted")
                    st.session_state.event_log.append(
                        {
                            "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                            "Event": "⚠️ DISTRACTED",
                            "Details": distract_res["distraction_reason"],
                        }
                    )
            else:
                current_state = "ALERT"

        else:
            # Face Not Found
            drowsy_engine.update_face_lost()
            distract_res = distraction_engine.update(
                face_found=False,
                pose_state="No Face",
                pitch=0.0,
                yaw=0.0,
                roll=0.0,
                mar=0.0,
                mar_threshold=active_mar_thresh,
                is_talking_mic=st.session_state.mic_detector.is_talking,
                multiple_faces=False,
                fps=fps,
            )
            drowsy_res = {
                "is_drowsy": False,
                "eyes_closed": False,
                "smoothed_ear": 0.0,
                "blink_count": drowsy_engine.blink_count,
                "drowsy_duration": 0.0,
                "total_drowsy_duration": drowsy_engine.total_drowsy_duration,
            }
            expression = "None"
            current_state = "DISTRACTED"
            pitch_adj = 0.0
            yaw_adj = 0.0
            pose_state_adj = "No Face"

        # 8. Attention Scorer Calculation
        attn_score = attention_scorer.calculate_score(
            is_drowsy=drowsy_res["is_drowsy"],
            is_distracted=distract_res["is_distracted"],
            is_yawning=distract_res["is_yawning"],
            eyes_closed=drowsy_res.get("eyes_closed", False),
            pose_state=pose_state_adj,
            pitch=pitch_adj,
            yaw=yaw_adj,
            avg_ear=features["avg_ear"],
            ear_threshold=active_ear_thresh,
        )

        # 9. Session Analytics Recording
        timestamp_str = datetime.datetime.now().strftime("%H:%M:%S")
        analytics.record_frame(
            time.time(),
            features["avg_ear"],
            features["mar"],
            attn_score,
            current_state,
        )

        # 10. CSV Telemetry Exporting
        exporter.log_frame(
            timestamp_str,
            features["avg_ear"],
            features["mar"],
            yaw_adj,
            pitch_adj,
            pose_data["roll"],
            current_state,
            expression,
            drowsy_res["blink_count"],
            distract_res["yawn_count"],
            attn_score,
        )

        # 11. Render Annotated Video Frame (EVERY FRAME for 30 FPS video)
        annotated_frame = draw_visual_annotations(
            enhanced_frame,
            mesh_data,
            features,
            pose_data,
            current_state,
            is_low_light,
            calibrator.is_calibrating,
            calib_progress,
        )

        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Update Telemetry Metrics & Status Pills (Throttled every 3 frames for zero lag)
        if frame_counter % 3 == 0 or current_state != prev_state_logged:
            prev_state_logged = current_state
            with status_placeholder.container():
                render_status_pill(current_state)
                st.markdown(
                    f"""
                    <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; margin-bottom: 15px;">
                        <div style="background: rgba(255, 255, 255, 0.08); backdrop-filter: blur(10px); padding: 6px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); color: #f5f5f5; font-size: 13px; font-weight: 500;">
                            🎭 Expression: <strong style="color: #00ff87;">{expression}</strong>
                        </div>
                        <div style="background: rgba(255, 255, 255, 0.08); backdrop-filter: blur(10px); padding: 6px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); color: #f5f5f5; font-size: 13px; font-weight: 500;">
                            🎙️ Sound Level: <strong style="color: #00d7ff;">{st.session_state.mic_detector.current_db:.1f} dB</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            score_metric.metric("Attention Score", f"{attn_score:.1f} / 100")
            ear_metric.metric("Avg EAR", f"{features['avg_ear']:.3f}")
            blink_metric.metric("Blink Count", f"{drowsy_res['blink_count']}")
            drowsy_dur_metric.metric("Drowsy Time", f"{drowsy_res['total_drowsy_duration']:.1f} s")

            pose_metric.metric("Head Pose", f"{pose_state_adj}")
            mar_metric.metric("MAR (Mouth)", f"{features['mar']:.3f}")
            yawn_metric.metric("Yawn Count", f"{distract_res['yawn_count']}")
            distraction_metric.metric("Distraction Count", f"{distract_res['distraction_count']}")

            session_dur = analytics.get_session_duration()
            summary_stats = analytics.compute_summary(
                drowsy_res["blink_count"],
                distract_res["yawn_count"],
                drowsy_res["total_drowsy_duration"],
                distract_res["total_distraction_duration"],
                distract_res["longest_distraction"],
                drowsy_engine.drowsy_event_count,
                distract_res["distraction_count"],
            )

            session_time_metric.metric("Session Time", f"{session_dur:.1f} s")
            focus_pct_metric.metric("Focus %", f"{summary_stats['focus_percentage']:.1f} %")

        # Update Heavy Components (Timeline Chart & Event Log DataFrame throttled every 10 frames)
        if frame_counter % 10 == 0:
            if len(analytics.history_timestamps) > 2:
                chart_df = pd.DataFrame(
                    {
                        "EAR (Eye)": analytics.history_ear[-60:],
                        "MAR (Mouth)": analytics.history_mar[-60:],
                    }
                )
                chart_placeholder.line_chart(chart_df, height=180)

            if st.session_state.event_log:
                event_df = pd.DataFrame(st.session_state.event_log)
                event_log_placeholder.dataframe(
                    event_df.tail(8),
                    use_container_width=True,
                    hide_index=True,
                )

        # Sleep minimal time to yield execution thread
        time.sleep(0.001)

    if cap is not None:
        cap.release()
    st.session_state.mic_detector.stop()

    # Session Summary Display when Stopped
    st.markdown("---")
    st.markdown("## 📈 Session Summary & Performance Analysis")

    summary = analytics.compute_summary(
        drowsy_engine.blink_count,
        distraction_engine.yawn_count,
        drowsy_engine.total_drowsy_duration,
        distraction_engine.total_distraction_duration,
        distraction_engine.longest_distraction,
        drowsy_engine.drowsy_event_count,
        distraction_engine.distraction_count,
    )

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total Blinks", summary["total_blinks"])
    sc2.metric("Total Yawns", summary["total_yawns"])
    sc3.metric("Focus Percentage", f"{summary['focus_percentage']}%")
    sc4.metric("Risk Level", summary["risk_level"])

    st.markdown("#### 🤖 AI Study Recommendations")
    for rec in summary["recommendations"]:
        st.info(rec)

    # CSV Download Buttons (Frame Telemetry & Event Log)
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_df = exporter.get_csv_dataframe()
        if not csv_df.empty:
            csv_data = csv_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Detailed CSV Telemetry Report",
                data=csv_data,
                file_name=exporter.csv_filename,
                mime="text/csv",
                use_container_width=True,
                key="summary_telemetry_csv_btn",
            )
    with col_dl2:
        if st.session_state.event_log:
            event_df = pd.DataFrame(st.session_state.event_log)
            event_csv_data = event_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📜 Download Session Event Log (CSV)",
                data=event_csv_data,
                file_name=f"session_event_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="summary_event_log_csv_btn",
            )
