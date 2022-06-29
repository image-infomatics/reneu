[![Build Status](https://travis-ci.org/jingpengw/reneu.svg?branch=master)](https://travis-ci.org/jingpengw/reneu)
![Python Package](https://github.com/jingpengw/reneu/workflows/Python%20package/badge.svg)


> :warning: **This package is still under development, it was customized to process a Electron Microscopy volume. The Software is provided "as is" without warranty of any kind, either express or implied. Use at your own risk.**

Computation for REal NEUral networks

# Features
- [x] Watershed
- [x] Greedy mean-affinity agglomeration
- [ ] Mutex watershed
- [ ] Segmentation evaluation
    - [ ] Rand error
    - [ ] Variation of information
    - [ ] Average run length
- [x] NBLAST algorithm to compare neuron skeletons. 

# Development

GCC version: 11.2.0

install dependencies

    conda env create -f environment.yml

if you use OSX, please use the mac-environment.yml.

compile C++ backend

    python setup.py develop

export conda environment

    conda env export > environment.yml

## Debug
build with debug mode:

    python setup.py build --debug

insert a breakpoint in python code before using the c++ module

run the test script

launch the debuger in VSCode and find the python process. 

continue the debugging python process, now we can use the debuger in VSCode.

## Release

```
python setup.py bdist_wheel
auditwheel repair dist/reneu-0.0.1-cp37-cp37m-linux_x86_64.whl
twine upload 
```
