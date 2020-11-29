#!/bin/bash

rm -rf ./build

python setup.py develop

pytest tests/test_dendrogram.py -s

