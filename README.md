[![Build Status](https://travis-ci.org/jingpengw/reneu.svg?branch=master)](https://travis-ci.org/jingpengw/reneu)
Computation for REal NEUral networks

# Development

install boost library (example for ubuntu):
    sudo apt install libboost-dev

compile C++ backend
    python setup.py develop

export conda environment

    conda env export > environment.yml

## Release

```
python setup.py bdist_wheel
auditwheel repair dist/reneu-0.0.1-cp37-cp37m-linux_x86_64.whl
twine upload 
```
