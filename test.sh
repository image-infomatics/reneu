#!/bin/bash

rm -rf ./build
#mv python/reneu/lib/segmentation.so /tmp/
rm python/reneu/lib/*.so 

#python setup.py develop
python setup.py build --debug

pytest tests/test_dendrogram.py -s
pytest tests/test_region_graph.py -s
pytest tests/test_region_graph_chunk.py -s

