"""
Feature extraction utilities for vehicle detection (HOG, spatial binning, color histogram).
"""

import cv2
import numpy as np
from skimage.feature import hog


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """Extract Histogram of Oriented Gradients (HOG) features from a single-channel image."""
    if vis:
        features, hog_image = hog(
            img,
            orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            transform_sqrt=True,
            visualize=True,
            feature_vector=feature_vec,
        )
        return features, hog_image

    features = hog(
        img,
        orientations=orient,
        pixels_per_cell=(pix_per_cell, pix_per_cell),
        cells_per_block=(cell_per_block, cell_per_block),
        transform_sqrt=True,
        visualize=False,
        feature_vector=feature_vec,
    )
    return features


def bin_spatial(img, size=(32, 32)):
    """Compute binned color features by resizing and raveling."""
    return cv2.resize(img, size).ravel()


def color_hist(img, nbins=32, bins_range=(0, 256)):
    """Compute color histogram features across all 3 channels."""
    ch1 = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)[0]
    ch2 = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)[0]
    ch3 = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)[0]
    return np.concatenate((ch1, ch2, ch3))


def _convert_color(image, color_space):
    """Convert an RGB image to the specified color space."""
    conversions = {
        "HSV": cv2.COLOR_RGB2HSV,
        "LUV": cv2.COLOR_RGB2LUV,
        "HLS": cv2.COLOR_RGB2HLS,
        "YUV": cv2.COLOR_RGB2YUV,
        "YCrCb": cv2.COLOR_RGB2YCrCb,
    }
    if color_space in conversions:
        return cv2.cvtColor(image, conversions[color_space])
    return np.copy(image)


def extract_features(imgs, color_space="RGB", spatial_size=(32, 32),
                     hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2,
                     hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Extract feature vectors from a list of image file paths.

    Returns a list of concatenated feature vectors (one per image).
    """
    features = []
    for filepath in imgs:
        file_features = []
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        feature_image = _convert_color(image, color_space)

        if spatial_feat:
            file_features.append(bin_spatial(feature_image, size=spatial_size))

        if hist_feat:
            file_features.append(color_hist(feature_image, nbins=hist_bins))

        if hog_feat:
            if hog_channel == "ALL":
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(
                        feature_image[:, :, channel], orient, pix_per_cell,
                        cell_per_block, vis=False, feature_vec=True,
                    ))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(
                    feature_image[:, :, hog_channel], orient, pix_per_cell,
                    cell_per_block, vis=False, feature_vec=True,
                )
            file_features.append(hog_features)

        features.append(np.concatenate(file_features))
    return features


def single_img_features(img, color_space="RGB", spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256), orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """Extract the same feature set from a single in-memory RGB image."""
    img_features = []
    feature_image = _convert_color(img, color_space)

    if spatial_feat:
        img_features.append(bin_spatial(feature_image, size=spatial_size))

    if hist_feat:
        img_features.append(color_hist(feature_image, nbins=hist_bins, bins_range=hist_range))

    if hog_feat:
        if hog_channel == "ALL":
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(
                    feature_image[:, :, channel], orient, pix_per_cell,
                    cell_per_block, vis=False, feature_vec=True,
                ))
        else:
            hog_features = get_hog_features(
                feature_image[:, :, hog_channel], orient, pix_per_cell,
                cell_per_block, vis=False, feature_vec=True,
            )
        img_features.append(np.array(hog_features))

    return np.concatenate(img_features)
