# Bag Frame Synchronization

## Overview

Two-node system for labeling cone clusters with synchronized bag frames:

1. **cluster_labeler.py** - Labels clusters and publishes timestamp info
2. **bag_frame_publisher.py** - Reads mcap bag and publishes synced frames

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/labeling/current_cluster` | PointCloud2 | Current cluster being labeled |
| `/labeling/current_timestamp` | String | JSON with timestamp/frame info |
| `/labeling/synced_frame` | PointCloud2 | Full bag frame matching cluster timestamp |

## Usage

### Terminal 1: Start bag frame publisher
```bash
cd /home/praksz/FRT2026/Cone-Labeler/Cone-Cluster-Labeler
. /opt/ros/jazzy/setup.bash
# Use ROS Python; add project venv site-packages for MCAP
PYTHONPATH="$PWD/cone_detector_env/lib/python3.12/site-packages:$PYTHONPATH" \
	python3 bag_frame_publisher.py
```

### Terminal 2: Start cluster labeler
```bash
cd /home/praksz/FRT2026/Cone-Labeler/Cone-Cluster-Labeler
. /opt/ros/jazzy/setup.bash
python3 cluster_labeler.py
```

### Foxglove
Open Foxglove and add:
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
