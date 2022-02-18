#!/bin/bash 

sudo apt update
sudo apt install build-essential
sudo apt install libboost-dev
python -m pip install -U pip
pip install -r requirements.txt
