#!/bin/bash

# 10.3.0 used to work to compile reneu
#module load gcc/10.3.0


module load gcc/11.2.0
module load cmake 

#rm -rf ./build
#mv python/reneu/lib/segmentation.so /tmp/
rm python/reneu/lib/skeleton.so 
rm python/reneu/lib/segmentation.so 

#python setup.py develop
python setup.py build --debug

#pytest tests/segmentation/test_disjoint_sets.py -s --pdb
#pytest tests/segmentation/test_dendrogram.py -s --pdb
#pytest tests/segmentation/test_region_graph.py -s --pdb
#pytest tests/segmentation/test_region_graph_chunk.py -s --pdb

