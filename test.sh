#!/bin/bash

rm -rf ./build
#mv python/reneu/lib/segmentation.so /tmp/
rm python/reneu/lib/*.so 

#python setup.py develop
python setup.py build --debug

pytest tests/segmentation/test_disjoint_sets.py -s --pdb
pytest tests/segmentation/test_dendrogram.py -s --pdb
pytest tests/segmentation/test_region_graph.py -s --pdb
pytest tests/segmentation/test_region_graph_chunk.py -s --pdb

