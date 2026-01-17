#!/bin/bash
set -e

echo "Setting up Cone Cluster Labeler environment..."

# Ensure ROS is available
source /opt/ros/jazzy/setup.bash

# Remove old broken venv if it exists
if [ -d "cluster_labeler_env" ]; then
  echo "Removing existing virtual environment..."
  rm -rf cluster_labeler_env
fi

# Create venv WITH system packages and correct Python
python3.12 -m venv --system-site-packages cluster_labeler_env

# Activate it
source cluster_labeler_env/bin/activate

# Upgrade pip (safe)
python -m pip install --upgrade pip

# Install Python-only dependencies
pip install -r requirements.txt

echo "✅ Environment setup complete"
echo "👉 Activate with: source cluster_labeler_env/bin/activate"
