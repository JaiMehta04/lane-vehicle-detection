"""
Shared image processing utility functions.
"""

import cv2
import numpy as np


def grayscale(img):
    """Convert an image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def gaussian_blur(img, kernel_size):
    """Apply Gaussian blur to an image."""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    """Apply Canny edge detection."""
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, vertices):
    """
    Apply an image mask. Only keeps the region defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)


def draw_lines(img, lines, color=(255, 0, 0), thickness=2):
    """Draw lines on an image in-place."""
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def weighted_img(img, initial_img, alpha=0.8, beta=1.0, gamma=0.0):
    """Overlay `img` (lines) onto `initial_img` with blending weights."""
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """Draw bounding boxes on a copy of the image."""
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy
