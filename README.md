# Cone Cluster Labeler & Random Forest Trainer

Cone detection and color classification system for Formula Student using LiDAR point cloud data.

---

## Overview

Two or three (only with color classification) node system for labeling cone clusters with synchronized bag frames:

1. **label.py** - Labels clusters and publishes timestamp info
2. **bag_frame_publisher.py** - Reads mcap bag and publishes synced frames

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/labeling/current_cluster` | PointCloud2 | Current cluster being labeled |
| `/labeling/current_timestamp` | String | JSON with timestamp/frame info |
| `/labeling/synced_frame` | PointCloud2 | Full bag frame matching cluster timestamp |

## Usage

Copy the .mcap bag to the Dataset folder.

### Terminal 1: Start bag frame publisher
```bash
# ROS2 Jazzy required
. /opt/ros/jazzy/setup.bash
pip install mcap mcap-ros2-support
```

---

## 🏷️ Labeling Process

### Starting the Labeler

**Terminal 1: Bag frame publisher**
```bash
. /opt/ros/jazzy/setup.bash
python3 label/bag_frame_publisher.py
```

**Terminal 2: Cluster labeler**
```bash
. /opt/ros/jazzy/setup.bash
python3 label/label.py
```
### Keyboard Controls for labeling
| Key | Function |
|-----|----------|
| `b` | Blue cone |
| `Y` | Yellow cone |
| `o` | Orange cone |
| `u` | Unknown color (but is a cone) |
| `n` | Not a cone |
| `s` | Skip |
| `q` | Quit (saves automatically) |

**Terminal 3: Image Finder**
Only needed for color classification
```bash
. /opt/ros/jazzy/setup.bash
python3 image_finder.py
```

### Foxglove Setup
Open your Foxglove bridge, or open Foxglove and add:
- **3D Panel**: Subscribe to `/labeling/current_cluster` (colored cluster)
- **3D Panel**: Subscribe to `/labeling/synced_frame` (full scene from bag)


## How It Works

1. Cluster labeler extracts timestamp from filename: `scan_<TIMESTAMP>_frame_<NUM>_cluster_<ID>.pcd`
2. Publishes timestamp as JSON: `{"scan_timestamp": "...", "frame_number": "...", "cluster_id": "..."}`
3. Bag publisher receives timestamp → finds matching frame in mcap → publishes full pointcloud
4. Both pointclouds visible in Foxglove for context

## Dependencies

Install MCAP libraries (choose one option):
```bash
# Option A: Install into ROS Python (system interpreter)
. /opt/ros/jazzy/setup.bash
python3 -m pip install mcap mcap-ros2-support --break-system-packages

# Option B: Install into project venv (used via PYTHONPATH)
./cone_detector_env/bin/python -m pip install mcap mcap-ros2-support --break-system-packages
```

## 🔧 Utilities

### Frame Filter
Reduces frame count for faster labeling:
```bash
python filter_frames.py
```

### Model Testing  TEST_NEW_MODEL?
```bash
cd test
python test_models.py
```

## 🤖 Model Training

### Detection Model (cone/not-cone)
```bash
python train.py
```

**Extracted Features:**
- Height, width, depth
- Aspect ratio
- Point density
- Volume
- Distance from LiDAR

### Color Model (blue/yellow/orange)
```bash
python train_color.py
```

**Extracted Features:**
- Normalized intensity mean/std (distance-dependent)
- Vertical intensity gradient
- Height and aspect ratios
- Reflective point percentage
- Bottom/middle zone intensity contrast

---

## 📊 Training Output & Logging

The training scripts automatically generate logs, figures, and saves the trained models.


### Training Logs
Every training run is logged to `logs/detection/` or `logs/color/`:
- Training parameters and hyperparameters
- Dataset statistics (sample count, class distribution)
- Cross-validation results
- Final model performance metrics (accuracy, precision, recall, F1-score)

### Generated Figures
Saved to `figures/detection/` or `figures/color/`:

| Figure | Description |
|--------|-------------|
| **Confusion Matrix** | Visualizes prediction accuracy per class, helps identify misclassification patterns |
| **Feature Importance** | Bar chart showing which features contribute most to predictions |
| **Correlation Heatmap** | Shows relationships between extracted features, useful for feature engineering |

### Saved Models
Models are serialized to `models/` folder as `.pkl` files:
- `models/detection/cone_detector_rf.pkl` - Detection classifier
- `models/color/cone_color_classifier_rf.pkl` - Color classifier
- Includes trained model + fitted scaler for inference

