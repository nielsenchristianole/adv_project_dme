#!/bin/bash

# Visualize
python src/data/undistort-map/project_map.py --inp_npy coordinates.npy --output_dir output --visualize --radius 1 --planes_file test

# Don't visualize
#python src/data/undistort-map/project_map.py --inp_npy coordinates.npy --output_dir output --no-visualize --radius 1 --planes_file test
