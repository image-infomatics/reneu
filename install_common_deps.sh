#!/bin/bash 

sudo apt update -y
sudo apt install build-essential
sudo apt install cmake xtensor-dev xtensor-python-dev libboost-dev
python -m pip install -U pip
pip install -r requirements.txt
