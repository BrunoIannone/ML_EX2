#!/bin/bash



# Set the name of your conda environment
ENV_NAME=ML_EX2

# Set the path to your requirements.txt file
REQUIREMENTS_FILE=requirements.txt

# Set the Python version
PYTHON_VERSION=3.11



# Create and activate the conda environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
conda activate $ENV_NAME

# Install gymnasium
pip install gymnasium[box2d]

# Install requirements from requirements.txt
pip install -r $REQUIREMENTS_FILE -y

echo "Conda environment with Python $PYTHON_VERSION created and activated."
echo "Requirements installed from requirements.txt."

