#!/bin/bash
echo "Setting up the environment..."

python3 -m venv cluster_labeler_env
source cluster_labeler_env/bin/activate

pip install -r requirements.txt

echo "Setup compplete"