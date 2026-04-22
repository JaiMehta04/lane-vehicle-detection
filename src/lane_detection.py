"""
Lane detection pipeline using Canny edge detection and Hough transforms.
"""

import numpy as np
import cv2

from src.utils import grayscale, gaussian_blur, canny, region_of_interest, draw_lines, weighted_img
from config import LANE_PARAMS


def _separate_lines(lines):
    """Separate Hough lines into right and left groups by slope."""
    right, left = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        m = (float(y2) - y1) / (x2 - x1)
        if m >= 0:
            right.append([x1, y1, x2, y2, m])
        else:
            left.append([x1, y1, x2, y2, m])
    return right, left


def _extend_point(x1, y1, x2, y2, length):
    """Extend a line segment by a given length from the (x2, y2) endpoint."""
    line_len = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    x = x2 + (x2 - x1) / line_len * length
    y = y2 + (y2 - y1) / line_len * length
    return x, y


def _filter_outliers(data, cutoff, thresh=0.08):
    """Keep only lines whose slopes fall within cutoff range and near the mean."""
    if len(data) == 0:
        return np.array([])
    data = np.array(data)
    data = data[(data[:, 4] >= cutoff[0]) & (data[:, 4] <= cutoff[1])]
    if len(data) == 0:
        return data
    m = np.mean(data[:, 4], axis=0)
    return data[(data[:, 4] <= m + thresh) & (data[:, 4] >= m - thresh)]


def _merge_lines(lines):
    """Average line segments and extend them across the image. Returns None if empty."""
    if len(lines) == 0:
        return None
    lines = np.array(lines)[:, :4]
    x1, y1, x2, y2 = np.mean(lines, axis=0)
    if np.isnan(x1):
        return None
    x1e, y1e = _extend_point(x1, y1, x2, y2, -1000)
    x2e, y2e = _extend_point(x1, y1, x2, y2, 1000)
    return np.array([[[x1e, y1e, x2e, y2e]]], dtype=np.int32)


def detect_lanes(image):
    """
    Run the full lane detection pipeline on a single frame.

    Parameters
    ----------
    image : np.ndarray
        RGB image (H x W x 3).

    Returns
    -------
    np.ndarray
        The original image with detected lane lines overlaid.
    """
    p = LANE_PARAMS
    vertices = [np.array(
        [p["bot_left"], p["bot_right"], p["apex_right"], p["apex_left"]],
        dtype=np.int32,
    )]

    # Pre-processing
    gray = grayscale(image)
    blur = gaussian_blur(gray, p["blur_kernel"])
    edge = canny(blur, p["canny_low"], p["canny_high"])
    mask = region_of_interest(edge, vertices)

    # Hough line detection
    lines = cv2.HoughLinesP(
        mask,
        p["hough_rho"],
        np.pi / p["hough_theta_divisor"],
        p["hough_threshold"],
        np.array([]),
        minLineLength=p["hough_min_line_len"],
        maxLineGap=p["hough_max_line_gap"],
    )

    if lines is None:
        return image

    # Separate, filter, and merge lines
    right_lines, left_lines = _separate_lines(lines)

    right = _filter_outliers(right_lines, cutoff=p["right_slope_range"])
    right = _merge_lines(right)

    left = _filter_outliers(left_lines, cutoff=p["left_slope_range"])
    left = _merge_lines(left)

    merged = [l for l in (right, left) if l is not None]
    if not merged:
        return image
    merged = np.concatenate(merged)

    # Draw on blank canvas, then blend
    line_img = np.zeros_like(image)
    draw_lines(line_img, merged, thickness=p["line_thickness"])
    line_img = region_of_interest(line_img, vertices)

    return weighted_img(line_img, image)
