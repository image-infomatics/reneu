#!/bin/bash

rm -rf ./build
mv python/reneu/lib/*.so /tmp/

python setup.py develop

pytest tests/test_dendrogram.py -s
pytest tests/test_region_graph.py -s

