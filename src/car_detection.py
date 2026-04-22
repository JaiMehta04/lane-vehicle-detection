"""
Vehicle detection pipeline using a trained SVM classifier, sliding windows,
heatmaps, and frame-to-frame tracking.
"""

import pickle

import cv2
import numpy as np
from scipy import ndimage as ndi

from src.feature_extraction import single_img_features
from src.utils import draw_boxes
from config import FEATURE_PARAMS, WINDOW_SEARCH, MODEL_PATH


# ──────────────────────────────────────────────
# Sliding Window
# ──────────────────────────────────────────────

def slide_window(img, x_start_stop=None, y_start_stop=None,
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """Generate a list of sliding window positions across a region of the image."""
    if x_start_stop is None:
        x_start_stop = [0, img.shape[1]]
    if y_start_stop is None:
        y_start_stop = [0, img.shape[0]]

    x_start = x_start_stop[0] or 0
    x_stop = x_start_stop[1] or img.shape[1]
    y_start = y_start_stop[0] or 0
    y_stop = y_start_stop[1] or img.shape[0]

    nx_pix_per_step = int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = int(xy_window[1] * (1 - xy_overlap[1]))

    nx_windows = int((x_stop - x_start) // nx_pix_per_step) - 1
    ny_windows = int((y_stop - y_start) // ny_pix_per_step) - 1

    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs * nx_pix_per_step + x_start
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start
            endy = starty + xy_window[1]
            window_list.append(((startx, starty), (endx, endy)))
    return window_list


# ──────────────────────────────────────────────
# Search Windows with Classifier
# ──────────────────────────────────────────────

def search_windows(img, windows, clf, scaler, params):
    """Classify each window and return those predicted as vehicles."""
    on_windows = []
    for window in windows:
        test_img = cv2.resize(
            img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64)
        )
        features = single_img_features(
            test_img,
            color_space=params["color_space"],
            spatial_size=params["spatial_size"],
            hist_bins=params["hist_bins"],
            orient=params["orient"],
            pix_per_cell=params["pix_per_cell"],
            cell_per_block=params["cell_per_block"],
            hog_channel=params["hog_channel"],
            spatial_feat=params["spatial_feat"],
            hist_feat=params["hist_feat"],
            hog_feat=params["hog_feat"],
        )
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        if clf.predict(test_features) == 1:
            on_windows.append(window)
    return on_windows


# ──────────────────────────────────────────────
# Heatmap and Box Combination
# ──────────────────────────────────────────────

def _create_heatmap(windows, image_shape):
    """Build a heatmap from overlapping bounding boxes."""
    heatmap = np.zeros(image_shape[:2])
    for w in windows:
        heatmap[w[0][1]:w[1][1], w[0][0]:w[1][0]] += 1
    return heatmap


def _find_windows_from_heatmap(heatmap, threshold=0):
    """Label connected components in the heatmap and return bounding boxes."""
    heatmap[heatmap <= threshold] = 0
    labels = ndi.label(heatmap)
    boxes = []
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        boxes.append(((np.min(nonzerox), np.min(nonzeroy)),
                       (np.max(nonzerox), np.max(nonzeroy))))
    return boxes


def _combine_boxes(windows, image_shape):
    """Merge overlapping detection windows via heatmap."""
    if len(windows) == 0:
        return [], None
    heatmap = _create_heatmap(windows, image_shape)
    hot_windows = _find_windows_from_heatmap(heatmap)
    return hot_windows, heatmap


def _do_overlap(box1, box2):
    """Check whether two bounding boxes overlap."""
    l1x, l1y = box1[0]
    r1x, r1y = box1[1]
    l2x, l2y = box2[0]
    r2x, r2y = box2[1]
    if l1x > r2x or l2x > r1x:
        return False
    if l1y > r2y or l2y > r1y:
        return False
    return True


def _temporal_filter(new_boxes, old_boxes):
    """Keep only boxes that overlap with detections from the previous frame."""
    if not old_boxes:
        return new_boxes, new_boxes

    confirmed = []
    for new_box in new_boxes:
        if any(_do_overlap(new_box, old_box) for old_box in old_boxes):
            confirmed.append(new_box)
    return confirmed, new_boxes


# ──────────────────────────────────────────────
# Vehicle Detector (stateful, for video processing)
# ──────────────────────────────────────────────

class VehicleDetector:
    """
    Stateful vehicle detector that loads a trained SVM model and applies
    sliding-window search with temporal filtering across video frames.
    """

    def __init__(self, model_path=MODEL_PATH):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.clf = data["svc"]
        self.scaler = data["X_scaler"]
        self.params = data["parameters"]
        self._old_boxes = []

    def _build_windows(self, image):
        """Build the multi-scale sliding window list."""
        windows = []
        for cfg in WINDOW_SEARCH:
            windows += slide_window(
                image,
                x_start_stop=[None, None],
                y_start_stop=cfg["y_start_stop"],
                xy_window=cfg["xy_window"],
                xy_overlap=cfg["xy_overlap"],
            )
        return windows

    def detect(self, image):
        """
        Detect vehicles in a single frame.

        Parameters
        ----------
        image : np.ndarray
            RGB image (H x W x 3).

        Returns
        -------
        np.ndarray
            Image with bounding boxes drawn around detected vehicles.
        """
        windows = self._build_windows(image)
        hot_windows = search_windows(image, windows, self.clf, self.scaler, self.params)
        combined, _ = _combine_boxes(hot_windows, image.shape)
        confirmed, self._old_boxes = _temporal_filter(combined, self._old_boxes)
        return draw_boxes(image, confirmed, color=(0, 0, 255), thick=6)

    def reset(self):
        """Reset temporal state (call before processing a new video)."""
        self._old_boxes = []
