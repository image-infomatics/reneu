from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os
import sys
import re
import setuptools
from shutil import move
import numpy 


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PACKAGE_DIR, 'requirements.txt')) as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if not l.startswith('#')]

with open("README.md", "r") as fh:
    long_description = fh.read()

VERSIONFILE = os.path.join(PACKAGE_DIR, "python/reneu/__version__.py")
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." %
                       (VERSIONFILE, ))


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        'xiuli',
        ['src/main.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            numpy.get_include(),
        ],
        language='c++',
        extra_compile_args=[
            # use Og for gdb debug
            '-Og',
            #'-O3',
            '-ffast-math',
            # build with debug info
            # '-g'
            # this is not working
            #'-o reneu/xiuli.so'
        ],
        # monitor the code change and rebuild
        depends = ['src/*', 'include/xiuli/*'],
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag): return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')



class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        # link to cblas library to solve undefined symbol issue
        'unix': ['-lcblas',],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        # remove the compilation warning. 
        # This flag only works with C rather than C++
        self.compiler.compiler_so.remove('-Wstrict-prototypes')

        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        for arg in ext_modules[0].extra_compile_args:
            opts.append(arg)
        
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' %
                        self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' %
                        self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts

        super().build_extensions()


setup(
    name='reneu',
    description='computation for real neural networks.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    version=version,
    author='Jingpeng Wu',
    author_email='jingpeng.wu@gmail.com',
    packages=find_packages(exclude=['tests', 'bin', 'docker', 'kubernetes']),
    url='https://github.com/jingpengw/reneu',
    install_requires=requirements,
    tests_require=[
        'pytest',
    ],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires='>=3',
    zip_safe=False,
)

# The -o option of gcc is not working
# use code to move all the compiled so files to lib folder!
for file_name in os.listdir():
    if file_name.startswith("xiuli") and file_name.endswith(".so"):
        dst_file_name = os.path.join('python/reneu/', file_name)
        # using the full destination path will replace the so file
        # if it is already exist
        move(file_name, dst_file_name)
