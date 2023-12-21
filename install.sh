#!/bin/bash

# Set the path to your Anaconda or Miniconda installation
CONDA_PATH=/path/to/your/conda/installation

# Set the name of your conda environment
ENV_NAME=ML_EX2

# Set the path to your requirements.txt file
REQUIREMENTS_FILE=requirements.txt

# Set the Python version
PYTHON_VERSION=3.11

# Set the full path to conda executable
CONDA_EXE=$CONDA_PATH/bin/conda

# Create and activate the conda environment
conda create -y -n $ENV_NAME python=$PYTHON_VERSION
conda activate $ENV_NAME

# Install gymnasium
pip install gymnasium[box2d]

# Install requirements from requirements.txt
conda install -y --file $REQUIREMENTS_FILE

echo "Conda environment with Python $PYTHON_VERSION created and activated."
echo "Requirements installed from requirements.txt."

