"""
Train an SVM classifier for vehicle detection and save the model.

Usage:
    python -m src.train
"""

import os
import pickle
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.feature_extraction import extract_features
from config import (
    FEATURE_PARAMS, SVM_MAX_ITER, TEST_SPLIT_RATIO,
    CAR_DATA_DIR, NON_CAR_DATA_DIR, MODEL_PATH, FEATURES_PATH,
)


def _load_image_paths():
    """Walk the car / non-car dataset directories and collect .png paths."""
    cars, notcars = [], []
    for root, _, files in os.walk(CAR_DATA_DIR):
        for f in files:
            if f.endswith(".png"):
                cars.append(os.path.join(root, f))
    for root, _, files in os.walk(NON_CAR_DATA_DIR):
        for f in files:
            if f.endswith(".png"):
                notcars.append(os.path.join(root, f))
    print(f"Loaded {len(cars)} car images and {len(notcars)} non-car images.")
    return cars, notcars


def extract_and_save_features():
    """Extract features from training images and persist to disk."""
    print("[1/5] Reading image paths ...")
    cars, notcars = _load_image_paths()

    print("[2/5] Extracting features ...")
    car_features = extract_features(cars, **FEATURE_PARAMS)
    notcar_features = extract_features(notcars, **FEATURE_PARAMS)

    print("[3/5] Normalizing and splitting dataset ...")
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    scaler = StandardScaler().fit(X)
    scaled_X = scaler.transform(X)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=TEST_SPLIT_RATIO, random_state=rand_state
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    print("[4/5] Saving features ...")
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump({
            "train_dataset": X_train,
            "train_labels": y_train,
            "test_dataset": X_test,
            "test_labels": y_test,
            "X_scaler": scaler,
            "parameters": FEATURE_PARAMS,
        }, f, pickle.HIGHEST_PROTOCOL)

    print("[5/5] Feature extraction complete.")
    return X_train, X_test, y_train, y_test, scaler


def train_model():
    """Train a Linear SVM and save the model pickle."""
    # Extract features (or load if already saved)
    if os.path.isfile(FEATURES_PATH):
        print("Found cached features, loading ...")
        with open(FEATURES_PATH, "rb") as f:
            data = pickle.load(f)
        X_train = data["train_dataset"]
        X_test = data["test_dataset"]
        y_train = data["train_labels"]
        y_test = data["test_labels"]
        scaler = data["X_scaler"]
    else:
        X_train, X_test, y_train, y_test, scaler = extract_and_save_features()

    print("Training Linear SVM ...")
    svc = LinearSVC(max_iter=SVM_MAX_ITER)
    t0 = time.time()
    svc.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 2)

    accuracy = round(svc.score(X_test, y_test), 4)
    print(f"  Training time : {elapsed}s")
    print(f"  Test accuracy : {accuracy}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "svc": svc,
            "X_scaler": scaler,
            "parameters": FEATURE_PARAMS,
        }, f, pickle.HIGHEST_PROTOCOL)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
