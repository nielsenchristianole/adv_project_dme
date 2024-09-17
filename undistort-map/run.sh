#!/bin/bash

# Visualize
python undistort-map/project_map.py --inp_npy coordinates.npy --output_dir output --visualize --radius 1 --planes_file test

# Don't visualize
#python undistort-map/project_map.py --inp_npy coordinates.npy --output_dir output --no-visualize --radius 1 --planes_file test
