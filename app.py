"""
Lane & Vehicle Detection — Streamlit Web App

Interactive demo portal for uploading dash-cam images/videos and running
lane detection + vehicle detection with real-time metrics display.

Usage:
    streamlit run app.py
"""

import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from moviepy.editor import VideoFileClip
from PIL import Image

from src.lane_detection import detect_lanes
from src.advanced_lane_detection import AdvancedLaneDetector, compute_metrics
from src.car_detection import VehicleDetector
from config import MODEL_PATH, OUTPUT_DIR

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Lane & Vehicle Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a73e8 0%, #34a853 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a73e8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.3rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1a73e8, #34a853);
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar Controls
# ──────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/road.png", width=60)
    st.markdown("## Settings")

    mode = st.selectbox(
        "Detection Mode",
        ["Lane Detection", "Vehicle Detection", "Both (Lane + Vehicle)"],
        index=0,
        help="Choose what to detect in the uploaded file",
    )
    mode_map = {"Lane Detection": "lane", "Vehicle Detection": "car", "Both (Lane + Vehicle)": "both"}
    selected_mode = mode_map[mode]

    lane_mode = "advanced"
    if selected_mode in ("lane", "both"):
        lane_mode = st.radio(
            "Lane Detection Method",
            ["Advanced (Polynomial — handles curves)", "Basic (Hough — straight lines)"],
            index=0,
        )
        lane_mode = "advanced" if "Advanced" in lane_mode else "basic"

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This app detects **lane lines** and **vehicles** from dash-cam footage using "
        "computer vision and a trained SVM classifier. The advanced lane detector uses "
        "bird's-eye perspective warping and polynomial fitting to handle curved roads."
    )
    st.markdown(
        "**Tech:** Python · OpenCV · scikit-learn · Streamlit"
    )

# ──────────────────────────────────────────────
# Model Loading (cached)
# ──────────────────────────────────────────────

@st.cache_resource
def load_vehicle_detector():
    """Load the trained SVM vehicle detector (cached across reruns)."""
    if not os.path.isfile(MODEL_PATH):
        return None
    return VehicleDetector(MODEL_PATH)


# ──────────────────────────────────────────────
# Processing Helpers
# ──────────────────────────────────────────────

def process_frame(frame, selected_mode, lane_mode, lane_det, car_det):
    """Process a single frame through the selected pipeline(s)."""
    if selected_mode == "lane":
        if lane_mode == "advanced" and lane_det is not None:
            return lane_det.detect(frame)
        return detect_lanes(frame)
    elif selected_mode == "car":
        if car_det is not None:
            return car_det.detect(frame)
        return frame
    else:  # both
        if lane_mode == "advanced" and lane_det is not None:
            result = lane_det.detect(frame)
        else:
            result = detect_lanes(frame)
        if car_det is not None:
            result = car_det.detect(result)
        return result


def process_image_upload(uploaded_file, selected_mode, lane_mode):
    """Process an uploaded image and return (result_rgb, metrics_dict)."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    lane_det = AdvancedLaneDetector() if (lane_mode == "advanced" and selected_mode in ("lane", "both")) else None
    car_det = load_vehicle_detector() if selected_mode in ("car", "both") else None

    result = process_frame(img_rgb, selected_mode, lane_mode, lane_det, car_det)

    # Extract metrics if advanced lane detection was used
    metrics = None
    if lane_det is not None and lane_det._left_fit is not None and lane_det._right_fit is not None:
        metrics = compute_metrics(lane_det._left_fit, lane_det._right_fit, img_rgb.shape)

    return result, metrics


def process_video_upload(uploaded_file, selected_mode, lane_mode, progress_bar, status_text):
    """Process an uploaded video and return (output_path, final_metrics)."""
    # Save upload to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"web_result_{int(time.time())}.mp4")

    lane_det = AdvancedLaneDetector() if (lane_mode == "advanced" and selected_mode in ("lane", "both")) else None
    car_det = load_vehicle_detector() if selected_mode in ("car", "both") else None

    clip = VideoFileClip(tfile.name)
    total_frames = int(clip.fps * clip.duration)
    frame_count = [0]
    last_metrics = [None]

    def _process(frame):
        result = process_frame(frame, selected_mode, lane_mode, lane_det, car_det)
        frame_count[0] += 1
        pct = min(frame_count[0] / max(total_frames, 1), 1.0)
        progress_bar.progress(pct)
        status_text.text(f"Processing frame {frame_count[0]} / {total_frames}")

        if lane_det is not None and lane_det._left_fit is not None:
            last_metrics[0] = compute_metrics(lane_det._left_fit, lane_det._right_fit, frame.shape)

        return result

    result_clip = clip.fl_image(_process)
    result_clip.write_videofile(output_path, audio=False, logger=None)

    # Clean up temp file
    os.unlink(tfile.name)

    return output_path, last_metrics[0]


# ──────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────

st.markdown('<p class="main-header">Lane & Vehicle Detection System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    'Upload a dash-cam image or video to detect lane lines and vehicles with real-time metrics.'
    '</p>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"],
    help="Supported: JPG, PNG, BMP images and MP4, AVI, MOV videos",
)

if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    is_video = file_ext in (".mp4", ".avi", ".mov", ".mkv")

    # Warn if vehicle detection needed but no model
    if selected_mode in ("car", "both") and not os.path.isfile(MODEL_PATH):
        st.warning(
            "⚠️ Vehicle detection model not found. "
            "Run `python main.py --train` first to train the SVM classifier.",
            icon="⚠️",
        )
        if selected_mode == "car":
            st.stop()

    # ── IMAGE ──
    if not is_video:
        col_orig, col_result = st.columns(2)

        with col_orig:
            st.markdown("### 📷 Original")
            st.image(uploaded_file, use_container_width=True)

        with st.spinner("Running detection pipeline..."):
            uploaded_file.seek(0)
            result_img, metrics = process_image_upload(uploaded_file, selected_mode, lane_mode)

        with col_result:
            st.markdown("### 🎯 Detection Result")
            st.image(result_img, use_container_width=True)

        # Metrics cards
        if metrics:
            st.markdown("---")
            st.markdown("### 📊 Lane Metrics")
            m1, m2, m3 = st.columns(3)

            curv = metrics["avg_curvature_m"]
            offset = metrics["center_offset_m"]
            from config import ADV_LANE_PARAMS
            lane_w_m = metrics["lane_width_px"] * ADV_LANE_PARAMS["metrics"]["meters_per_pixel_x"]

            with m1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{"~Straight" if curv > 5000 else f"{curv:.0f} m"}</div>'
                    f'<div class="metric-label">Road Curvature Radius</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with m2:
                direction = "right" if offset > 0 else "left"
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{abs(offset):.2f} m</div>'
                    f'<div class="metric-label">Center Offset ({direction})</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with m3:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{lane_w_m:.2f} m</div>'
                    f'<div class="metric-label">Detected Lane Width</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Download button
        result_pil = Image.fromarray(result_img)
        import io
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download Result Image",
            data=buf.getvalue(),
            file_name="detection_result.png",
            mime="image/png",
        )

    # ── VIDEO ──
    else:
        st.markdown("### 🎬 Uploaded Video")
        st.video(uploaded_file)

        if st.button("🚀 Run Detection", type="primary", use_container_width=True):
            st.markdown("---")
            progress_bar = st.progress(0)
            status_text = st.empty()

            uploaded_file.seek(0)
            output_path, metrics = process_video_upload(
                uploaded_file, selected_mode, lane_mode, progress_bar, status_text,
            )

            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")

            st.markdown("### 🎯 Detection Result")
            st.video(output_path)

            # Metrics from last frame
            if metrics:
                st.markdown("### 📊 Lane Metrics (final frame)")
                m1, m2, m3 = st.columns(3)

                curv = metrics["avg_curvature_m"]
                offset = metrics["center_offset_m"]
                from config import ADV_LANE_PARAMS
                lane_w_m = metrics["lane_width_px"] * ADV_LANE_PARAMS["metrics"]["meters_per_pixel_x"]

                with m1:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">{"~Straight" if curv > 5000 else f"{curv:.0f} m"}</div>'
                        f'<div class="metric-label">Road Curvature Radius</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with m2:
                    direction = "right" if offset > 0 else "left"
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">{abs(offset):.2f} m</div>'
                        f'<div class="metric-label">Center Offset ({direction})</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with m3:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">{lane_w_m:.2f} m</div>'
                        f'<div class="metric-label">Detected Lane Width</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Download button
            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Result Video",
                    data=f.read(),
                    file_name="detection_result.mp4",
                    mime="video/mp4",
                )

else:
    # Landing state — show sample capabilities
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            "#### 🛣️ Lane Detection\n"
            "Detects curved and straight lane lines using bird's-eye perspective "
            "transform and polynomial fitting."
        )
    with col2:
        st.markdown(
            "#### 🚗 Vehicle Detection\n"
            "Identifies vehicles using HOG features and a trained Linear SVM "
            "with sliding-window search and heatmap filtering."
        )
    with col3:
        st.markdown(
            "#### 📊 Real-Time Metrics\n"
            "Displays road curvature, vehicle center offset, and lane width "
            "overlaid on every processed frame."
        )

    st.markdown("---")
    st.info("👆 Upload a dash-cam image or video above to get started!", icon="📤")
