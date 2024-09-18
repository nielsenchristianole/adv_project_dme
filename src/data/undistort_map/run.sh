#!/bin/bash

# Visualize
python src/data/undistort_map/project_map.py --inp_npy_file coordinates.npy --output_dir data/undistorted_output --visualize --radius 1 --planes_file test

# Don't visualize
#python src/data/undistort_map/project_map.py --inp_npy coordinates.npy --output_dir data/undistorted_output --no-visualize --radius 1 --planes_file test
