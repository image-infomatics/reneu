#!/bin/bash

rm -rf ./build

python setup.py develop

pytest tests/test_dendrogram.py -s
pytest tests/test_region_graph.py -s

