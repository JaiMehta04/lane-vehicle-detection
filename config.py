"""
Configuration parameters for lane detection and car detection pipelines.
"""

# ──────────────────────────────────────────────
# Lane Detection Parameters
# ──────────────────────────────────────────────

LANE_PARAMS = {
    # Region of interest vertices
    "bot_left": [80, 540],
    "bot_right": [980, 540],
    "apex_right": [510, 315],
    "apex_left": [450, 315],

    # Gaussian blur kernel size
    "blur_kernel": 7,

    # Canny edge detection thresholds
    "canny_low": 50,
    "canny_high": 125,

    # Hough transform parameters
    "hough_rho": 0.8,
    "hough_theta_divisor": 180,  # np.pi / this value
    "hough_threshold": 25,
    "hough_min_line_len": 50,
    "hough_max_line_gap": 200,

    # Slope cutoffs for line separation
    "right_slope_range": (0.45, 0.75),
    "left_slope_range": (-0.85, -0.6),

    # Line drawing thickness
    "line_thickness": 10,
}

# ──────────────────────────────────────────────
# Advanced Lane Detection Parameters
# ──────────────────────────────────────────────

ADV_LANE_PARAMS = {
    # Perspective transform source/destination (as fractions of image W, H)
    "perspective": {
        "src_top_left":  (0.44, 0.64),
        "src_top_right": (0.58, 0.64),
        "src_bot_right": (0.85, 0.95),
        "src_bot_left":  (0.20, 0.95),
        "dst_top_left":  (0.20, 0.0),
        "dst_top_right": (0.80, 0.0),
        "dst_bot_right": (0.80, 1.0),
        "dst_bot_left":  (0.20, 1.0),
    },
    # Thresholding
    "threshold": {
        "s_thresh": (120, 255),     # HLS S-channel
        "b_thresh": (155, 255),     # LAB B-channel
        "sobel_kernel": 3,
        "sobel_thresh": (20, 100),
    },
    # Sliding window search
    "sliding_window": {
        "n_windows": 9,
        "margin": 100,
        "min_pixels": 50,
    },
    # Real-world metrics (US highway standard lane ≈ 3.7m wide, dashed line ≈ 30m long)
    "metrics": {
        "meters_per_pixel_x": 3.7 / 700,
        "meters_per_pixel_y": 30.0 / 720,
    },
    # Smoothing factor for polynomial fit (0 = no smoothing, 1 = no memory)
    "smoothing_alpha": 0.3,
}

# ──────────────────────────────────────────────
# Car Detection / Feature Extraction Parameters
# ──────────────────────────────────────────────

FEATURE_PARAMS = {
    "color_space": "YCrCb",
    "orient": 8,
    "pix_per_cell": 8,
    "cell_per_block": 2,
    "hog_channel": "ALL",
    "spatial_size": (16, 16),
    "hist_bins": 32,
    "spatial_feat": True,
    "hist_feat": True,
    "hog_feat": True,
}

# ──────────────────────────────────────────────
# Sliding Window Search Parameters
# ──────────────────────────────────────────────

WINDOW_SEARCH = [
    {"y_start_stop": [400, 500], "xy_window": (96, 96), "xy_overlap": (0.75, 0.75)},
    {"y_start_stop": [400, 500], "xy_window": (144, 144), "xy_overlap": (0.75, 0.75)},
    {"y_start_stop": [430, 550], "xy_window": (192, 192), "xy_overlap": (0.75, 0.75)},
    {"y_start_stop": [460, 580], "xy_window": (192, 192), "xy_overlap": (0.75, 0.75)},
]

# ──────────────────────────────────────────────
# SVM Training Parameters
# ──────────────────────────────────────────────

SVM_MAX_ITER = 20000
TEST_SPLIT_RATIO = 0.2

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

DATA_DIR = "data"
MODEL_PATH = "models/model.p"
FEATURES_PATH = "models/features.p"
OUTPUT_DIR = "output"

CAR_DATA_DIR = "data/car-data/vehicles"
NON_CAR_DATA_DIR = "data/car-data/non-vehicles"
TEST_IMAGES_DIR = "data/test_images"
