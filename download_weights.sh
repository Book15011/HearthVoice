#!/bin/bash

# Create weights directory
mkdir -p gfpgan/weights

# Download GFPGAN weights
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P gfpgan/weights
wget https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth -P gfpgan/weights
wget https://github.com/xinntao/facexlib/releases/download/v0.1.0/parsing_parsenet.pth -P gfpgan/weights

echo "All weights downloaded successfully!"