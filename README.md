[![Build Status](https://travis-ci.org/jingpengw/reneu.svg?branch=master)](https://travis-ci.org/jingpengw/reneu)
![Python Package](https://github.com/jingpengw/reneu/workflows/Python%20package/badge.svg)

Computation for REal NEUral networks

# Development

compile C++ backend

    python setup.py develop

export conda environment

    conda env export > environment.yml

## Code Style

We follow the [Google Style](https://google.github.io/styleguide) for both [C++](https://google.github.io/styleguide/cppguide.html) and [Python](https://google.github.io/styleguide/pyguide.html).
We use [clang-format](https://clang.llvm.org/docs/ClangFormat.html) to format code automatically.

```
clang-format --style Google -i *.cc
```

## Release

```
python setup.py bdist_wheel
auditwheel repair dist/reneu-0.0.1-cp37-cp37m-linux_x86_64.whl
twine upload 
```
