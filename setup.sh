#!/bin/bash

# Clone the repository (This Repo is used for Depth Estimation)
git clone https://github.com/DepthAnything/Depth-Anything-V2.git

mv testing.py Depth-Anything-V2/
cd Depth-Anything-V2
pip install -r requirements.txt


