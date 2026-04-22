# Lane & Vehicle Detection System

A real-time computer vision system that detects **lane lines** and **vehicles** from dash-cam video footage using classical image processing techniques and machine learning.

<!--
<p align="center">
  <img src="data/laneLines_thirdPass.jpg" width="45%" />
  <img src="data/line-segments-example.jpg" width="45%" />
</p>
-->

## Highlights

- **Advanced Lane Detection** — Bird's-eye perspective warp → color/gradient thresholding → sliding-window search → 2nd-degree polynomial fitting for **curved lane lines**  
- **Real-Time Metrics HUD** — Displays radius of curvature, vehicle center offset, and lane width on every frame  
- **Vehicle Detection** — HOG + color histogram features → Linear SVM classifier → sliding-window search with heatmap-based deduplication  
- **Temporal Smoothing** — Exponential moving average on polynomial fits + frame-to-frame consistency checks for stable video output  
- **Modular Architecture** — Clean separation of lane detection, car detection, feature extraction, and training pipelines  
- **CLI Interface** — Single entry point to train, run basic or advanced lane detection, or full combined pipeline

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| Computer Vision | OpenCV, scikit-image |
| ML Classifier | scikit-learn (Linear SVM) |
| Video Processing | MoviePy |
| Feature Engineering | HOG, spatial binning, color histograms (YCrCb) |

## Project Structure

```
├── main.py                        # CLI entry point
├── app.py                         # Streamlit web app (interactive demo portal)
├── config.py                      # All tunable parameters and paths
├── src/
│   ├── advanced_lane_detection.py  # Curved-lane detection (polynomial + bird's-eye)
│   ├── lane_detection.py           # Basic straight-line detection (Hough)
│   ├── car_detection.py            # Vehicle detection (SVM + sliding window)
│   ├── feature_extraction.py       # HOG, spatial, color histogram features
│   ├── train.py                    # Model training script
│   └── utils.py                    # Shared image processing helpers
├── data/
│   ├── test_images/                # Sample images for testing
│   ├── car-data/                   # Training data (vehicles / non-vehicles)
│   └── *.mp4                       # Sample video clips
├── models/                         # Trained model artifacts (.p)
├── output/                         # Generated output videos/images
└── requirements.txt
```

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Train the Vehicle Detection Model

```bash
python main.py --train
```

This extracts HOG/color features from the training images in `data/car-data/`, trains a Linear SVM, and saves the model to `models/model.p`.

### Launch the Web App

```bash
streamlit run app.py
```

This opens an interactive demo portal in your browser where you can:
- Upload any dash-cam image or video
- Choose detection mode (Lane / Vehicle / Both)
- Toggle between basic and advanced lane detection
- View the processed result side-by-side with the original
- See real-time metrics (curvature, center offset, lane width)
- Download the processed result

### Run from CLI

**Both lane + vehicle detection (default — uses advanced lane detection):**
```bash
python main.py --input data/project_video.mp4
```

**Lane detection only (advanced — handles curves):**
```bash
python main.py --mode lane --input data/project_video.mp4
```

**Basic straight-line lane detection (Hough):**
```bash
python main.py --mode lane --lane-mode basic --input data/solidWhiteRight.mp4
```

**Vehicle detection only:**
```bash
python main.py --mode car --input data/project_video.mp4
```

**Custom output path:**
```bash
python main.py --mode both --input data/project_video.mp4 --output output/final_result.mp4
```

**Process a single image:**
```bash
python main.py --mode lane --input data/test_images/solidWhiteRight.jpg
```

## Pipeline Overview

### Advanced Lane Detection (default)
1. **Color thresholding** — HLS S-channel + LAB B-channel isolate white and yellow lane markings  
2. **Gradient thresholding** — Sobel-X operator detects vertical edges  
3. **Perspective warp** — Transform road to bird's-eye view (top-down)  
4. **Sliding window search** — Histogram-based starting points → window scan from bottom to top  
5. **Polynomial fitting** — Fit $ax^2 + bx + c$ to each lane's pixel coordinates  
6. **Sanity checks** — Validate lane width and parallel geometry, reject bad frames  
7. **Temporal smoothing** — Exponential moving average across frames for stability  
8. **Metrics computation** — Radius of curvature, vehicle offset from center, lane width  
9. **Inverse warp + overlay** — Project detected lanes back onto the original frame with a HUD  

### Basic Lane Detection (Hough)
1. Convert frame to grayscale → Gaussian blur → Canny edges  
2. Mask trapezoidal region of interest  
3. Hough Transform → separate lines by slope → filter outliers → merge  
4. Overlay straight lane lines on the original frame  

### Vehicle Detection
1. Slide multi-scale windows across the lower half of the frame  
2. For each window, extract features (HOG + spatial binning + color histograms)  
3. Classify each window with the trained SVM  
4. Build a heatmap from positive detections to merge overlapping boxes  
5. Apply temporal filtering — only keep detections consistent across frames  
6. Draw final bounding boxes on the frame  

## Metrics Displayed on Video

The advanced lane detector overlays a real-time **metrics HUD** on every frame:

| Metric | Description |
|--------|-------------|
| **Road Curvature** | Radius of curvature in meters (average of left + right lanes) |
| **Center Offset** | How far the vehicle is from lane center, in meters (left/right) |
| **Lane Width** | Detected lane width in meters |

## Configuration

All parameters are centralized in [`config.py`](config.py):

- **Lane detection**: ROI vertices, Canny thresholds, Hough parameters, slope cutoffs  
- **Advanced lane detection**: perspective warp points, color/S-channel thresholds, sliding-window params, smoothing alpha  
- **Feature extraction**: color space, HOG orientations, pixels per cell, spatial size  
- **Sliding window**: window sizes, overlap ratios, Y-axis search ranges  
- **Paths**: data directories, model save location, output folder  

## Sample Results

| Input | Lane Detection | Vehicle Detection |
|-------|---------------|-------------------|
| Dash-cam video | Curved lane overlay with green fill | Bounding boxes around vehicles |
| Dash-cam image | Lane lines + metrics HUD | Vehicle bounding boxes |

### Web App Screenshots

| Feature | Description |
|---------|-------------|
| Upload & detect | Drag-and-drop image/video upload with instant processing |
| Side-by-side view | Original vs. detected result comparison |
| Metrics cards | Road curvature, center offset, lane width displayed as cards |
| Video progress | Real-time progress bar during video processing |
| Download | One-click download of processed images and videos |

## License

This project is for educational and portfolio purposes.
