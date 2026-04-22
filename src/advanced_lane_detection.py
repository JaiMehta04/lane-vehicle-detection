"""
Advanced Lane Detection Pipeline

Uses perspective transform (bird's-eye view), color/gradient thresholding,
sliding-window search, and 2nd-degree polynomial fitting to detect curved lanes.
Computes real-world metrics: radius of curvature and vehicle offset from lane center.
"""

import cv2
import numpy as np

from config import ADV_LANE_PARAMS


# ──────────────────────────────────────────────
# Perspective Transform
# ──────────────────────────────────────────────

def _get_perspective_transforms(img_shape):
    """Return (warp_matrix, inverse_warp_matrix) for bird's-eye view."""
    h, w = img_shape[:2]
    p = ADV_LANE_PARAMS["perspective"]
    src = np.float32([
        [w * p["src_top_left"][0],  h * p["src_top_left"][1]],
        [w * p["src_top_right"][0], h * p["src_top_right"][1]],
        [w * p["src_bot_right"][0], h * p["src_bot_right"][1]],
        [w * p["src_bot_left"][0],  h * p["src_bot_left"][1]],
    ])
    dst = np.float32([
        [w * p["dst_top_left"][0],  h * p["dst_top_left"][1]],
        [w * p["dst_top_right"][0], h * p["dst_top_right"][1]],
        [w * p["dst_bot_right"][0], h * p["dst_bot_right"][1]],
        [w * p["dst_bot_left"][0],  h * p["dst_bot_left"][1]],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    return M, M_inv


def warp_image(img, M):
    """Apply perspective warp to an image."""
    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


# ──────────────────────────────────────────────
# Color & Gradient Thresholding
# ──────────────────────────────────────────────

def _abs_sobel_thresh(gray, orient="x", ksize=3, thresh=(0, 255)):
    """Apply Sobel operator and threshold the absolute gradient."""
    if orient == "x":
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobel = np.absolute(sobel)
    scaled = np.uint8(255 * abs_sobel / np.max(abs_sobel)) if np.max(abs_sobel) > 0 else np.zeros_like(abs_sobel, dtype=np.uint8)
    binary = np.zeros_like(scaled)
    binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return binary


def _color_threshold(img):
    """
    Combine HLS S-channel and LAB B-channel thresholds to robustly
    detect white and yellow lane lines under varying lighting.
    """
    p = ADV_LANE_PARAMS["threshold"]

    # HLS S-channel (good for yellow + white)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= p["s_thresh"][0]) & (s_channel <= p["s_thresh"][1])] = 1

    # LAB B-channel (good for yellow lines)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    b_channel = lab[:, :, 2]
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= p["b_thresh"][0]) & (b_channel <= p["b_thresh"][1])] = 1

    # Combine
    combined = np.zeros_like(s_binary)
    combined[(s_binary == 1) | (b_binary == 1)] = 1
    return combined


def create_binary(img):
    """
    Produce a binary image from combined color and gradient thresholds.
    """
    p = ADV_LANE_PARAMS["threshold"]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel X gradient
    sobel_binary = _abs_sobel_thresh(gray, orient="x", ksize=p["sobel_kernel"],
                                     thresh=p["sobel_thresh"])
    # Color thresholds
    color_binary = _color_threshold(img)

    # Merge: gradient OR color
    merged = np.zeros_like(gray)
    merged[(sobel_binary == 1) | (color_binary == 1)] = 1
    return merged


# ──────────────────────────────────────────────
# Sliding Window Lane Search
# ──────────────────────────────────────────────

def _sliding_window_search(binary_warped):
    """
    Find lane pixels using a sliding-window approach from the bottom up.
    Returns (left_fit, right_fit, left_x, left_y, right_x, right_y, out_img).
    """
    p = ADV_LANE_PARAMS["sliding_window"]
    nwindows = p["n_windows"]
    margin = p["margin"]
    minpix = p["min_pixels"]

    # Histogram of bottom half to find starting x positions
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = binary_warped.shape[0] // nwindows
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    # Visualisation image
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                       (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)

        if len(good_left) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left]))
        if len(good_right) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2) if len(lefty) > 0 else None
    right_fit = np.polyfit(righty, rightx, 2) if len(righty) > 0 else None

    return left_fit, right_fit, leftx, lefty, rightx, righty, out_img


def _search_around_poly(binary_warped, left_fit, right_fit):
    """
    Use prior polynomial fits to narrow the search area (faster than full sliding window).
    """
    margin = ADV_LANE_PARAMS["sliding_window"]["margin"]
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = (
        (nonzerox > (left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
        (nonzerox < (left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2] + margin))
    )
    right_lane_inds = (
        (nonzerox > (right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
        (nonzerox < (right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2] + margin))
    )

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit_new = np.polyfit(lefty, leftx, 2) if len(lefty) > 0 else left_fit
    right_fit_new = np.polyfit(righty, rightx, 2) if len(righty) > 0 else right_fit

    return left_fit_new, right_fit_new, leftx, lefty, rightx, righty


# ──────────────────────────────────────────────
# Metrics Computation
# ──────────────────────────────────────────────

def _compute_curvature(fit, y_eval, mx, my):
    """Compute radius of curvature in meters at y = y_eval."""
    A = fit[0] * mx / (my ** 2)
    B = fit[1] * mx / my
    return ((1 + (2 * A * y_eval * my + B) ** 2) ** 1.5) / abs(2 * A)


def compute_metrics(left_fit, right_fit, img_shape):
    """
    Compute lane metrics in real-world units.

    Returns dict with:
        - left_curvature_m   : radius of curvature of left lane (meters)
        - right_curvature_m  : radius of curvature of right lane (meters)
        - avg_curvature_m    : average curvature (meters)
        - center_offset_m    : vehicle offset from lane center (meters, +ve = right of center)
        - lane_width_px      : lane width in pixels at bottom of image
    """
    p = ADV_LANE_PARAMS["metrics"]
    mx = p["meters_per_pixel_x"]
    my = p["meters_per_pixel_y"]

    h = img_shape[0]
    y_eval = h - 1  # Evaluate curvature at bottom of image

    left_curv = _compute_curvature(left_fit, y_eval, mx, my)
    right_curv = _compute_curvature(right_fit, y_eval, mx, my)

    # Vehicle position: camera is assumed at image center
    left_x_bottom = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    right_x_bottom = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
    lane_center = (left_x_bottom + right_x_bottom) / 2
    image_center = img_shape[1] / 2
    offset = (image_center - lane_center) * mx

    return {
        "left_curvature_m": left_curv,
        "right_curvature_m": right_curv,
        "avg_curvature_m": (left_curv + right_curv) / 2,
        "center_offset_m": offset,
        "lane_width_px": right_x_bottom - left_x_bottom,
    }


# ──────────────────────────────────────────────
# Drawing & Overlay
# ──────────────────────────────────────────────

def _draw_lane_overlay(undistorted, binary_warped, left_fit, right_fit, M_inv):
    """Draw a green filled polygon for the detected lane and warp it back."""
    h, w = binary_warped.shape[:2]
    ploty = np.linspace(0, h - 1, h)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    color_warp = np.zeros((h, w, 3), dtype=np.uint8)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 200, 0))

    # Draw lane boundary lines
    cv2.polylines(color_warp, np.int_([pts_left]), False, (255, 255, 0), thickness=25)
    cv2.polylines(color_warp, np.int_([pts_right]), False, (255, 255, 0), thickness=25)

    # Warp back to original perspective
    newwarp = warp_image(color_warp, M_inv)
    return cv2.addWeighted(undistorted, 1, newwarp, 0.4, 0)


def _draw_metrics_overlay(image, metrics):
    """Render metrics as a HUD on the top-left of the frame."""
    curv = metrics["avg_curvature_m"]
    offset = metrics["center_offset_m"]
    direction = "right" if offset > 0 else "left"

    # Semi-transparent background panel
    overlay = image.copy()
    cv2.rectangle(overlay, (20, 15), (520, 145), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Curvature
    if curv > 5000:
        curv_text = "Road Curvature: ~Straight"
    else:
        curv_text = f"Road Curvature: {curv:.0f} m"
    cv2.putText(image, curv_text, (30, 50), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Center offset
    cv2.putText(image, f"Center Offset: {abs(offset):.2f} m {direction}",
                (30, 85), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Lane width
    lane_w_m = metrics["lane_width_px"] * ADV_LANE_PARAMS["metrics"]["meters_per_pixel_x"]
    cv2.putText(image, f"Lane Width: {lane_w_m:.2f} m",
                (30, 120), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return image


# ──────────────────────────────────────────────
# Stateful Lane Detector (for video)
# ──────────────────────────────────────────────

class AdvancedLaneDetector:
    """
    Stateful advanced lane detector that tracks polynomial fits across frames
    with exponential smoothing and confidence-based fallback.
    """

    def __init__(self):
        self._left_fit = None
        self._right_fit = None
        self._frame_count = 0
        self._fail_count = 0
        self._M = None
        self._M_inv = None
        self._smoothing = ADV_LANE_PARAMS.get("smoothing_alpha", 0.3)

    def reset(self):
        """Reset state for a new video."""
        self._left_fit = None
        self._right_fit = None
        self._frame_count = 0
        self._fail_count = 0
        self._M = None
        self._M_inv = None

    def _smooth_fit(self, new_fit, old_fit):
        """Exponential moving average of polynomial coefficients."""
        if old_fit is None:
            return new_fit
        a = self._smoothing
        return a * np.array(new_fit) + (1 - a) * np.array(old_fit)

    def _sanity_check(self, left_fit, right_fit, img_shape):
        """Validate that the two detected lanes make geometric sense."""
        h = img_shape[0]
        y_bottom = h - 1

        left_x = left_fit[0] * y_bottom ** 2 + left_fit[1] * y_bottom + left_fit[2]
        right_x = right_fit[0] * y_bottom ** 2 + right_fit[1] * y_bottom + right_fit[2]
        lane_width = right_x - left_x

        # Lane width should be roughly 500-900 pixels in warped image
        if lane_width < 300 or lane_width > 1100:
            return False

        # Check at top of image too
        y_top = 0
        left_x_top = left_fit[0] * y_top ** 2 + left_fit[1] * y_top + left_fit[2]
        right_x_top = right_fit[0] * y_top ** 2 + right_fit[1] * y_top + right_fit[2]
        top_width = right_x_top - left_x_top

        # Top width shouldn't be wildly different from bottom
        if top_width < 100 or abs(top_width - lane_width) > 600:
            return False

        return True

    def detect(self, image):
        """
        Detect curved lanes in a single frame with metrics overlay.

        Parameters
        ----------
        image : np.ndarray
            RGB image (H x W x 3).

        Returns
        -------
        np.ndarray
            Image with lane overlay and metrics HUD.
        """
        # Lazy-init perspective transform
        if self._M is None:
            self._M, self._M_inv = _get_perspective_transforms(image.shape)

        # 1. Create binary from color + gradient thresholds
        binary = create_binary(image)

        # 2. Warp to bird's-eye view
        binary_warped = warp_image(binary, self._M)

        # 3. Find lane pixels
        if self._left_fit is not None and self._right_fit is not None and self._fail_count < 5:
            left_fit, right_fit, lx, ly, rx, ry = _search_around_poly(
                binary_warped, self._left_fit, self._right_fit
            )
        else:
            left_fit, right_fit, lx, ly, rx, ry, _ = _sliding_window_search(binary_warped)

        # 4. Validate and smooth
        if left_fit is None or right_fit is None:
            self._fail_count += 1
            if self._left_fit is not None:
                return _draw_metrics_overlay(
                    _draw_lane_overlay(image, binary_warped, self._left_fit, self._right_fit, self._M_inv),
                    compute_metrics(self._left_fit, self._right_fit, image.shape),
                )
            return image

        if self._sanity_check(left_fit, right_fit, binary_warped.shape):
            left_fit = self._smooth_fit(left_fit, self._left_fit)
            right_fit = self._smooth_fit(right_fit, self._right_fit)
            self._left_fit = left_fit
            self._right_fit = right_fit
            self._fail_count = 0
        else:
            self._fail_count += 1
            if self._left_fit is not None:
                left_fit = self._left_fit
                right_fit = self._right_fit
            else:
                return image

        self._frame_count += 1

        # 5. Compute metrics
        metrics = compute_metrics(left_fit, right_fit, image.shape)

        # 6. Draw lane overlay + metrics HUD
        result = _draw_lane_overlay(image, binary_warped, left_fit, right_fit, self._M_inv)
        result = _draw_metrics_overlay(result, metrics)

        return result
