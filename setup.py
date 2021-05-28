import os
import re
import sys
import platform
import subprocess
from shutil import move 
from pathlib import Path

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


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


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.12.0':
                raise RuntimeError("CMake >= 3.12.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            '-DCMAKE_PREFIX_PATH=' + os.environ['CONDA_PREFIX'],
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', f'-j{os.cpu_count()}']

        env = os.environ.copy()
        env['LIBRARY_OUTPUT_DIRECTORY'] = self.build_temp
        env['CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG'] = self.build_temp
        env['CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE'] = self.build_temp

        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='reneu',
    description='computation for real neural networks.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    version=version,
    author='Jingpeng Wu',
    author_email='jingpeng.wu@gmail.com',
    packages=find_packages(exclude=['tests', 'build', 'reneu.egg-info', 'data']),
    url='https://github.com/jingpengw/reneu',
    install_requires=requirements,
    tests_require=[
        'pytest',
    ],
    ext_modules=[CMakeExtension('libreneu', sourcedir='./cpp')],
    cmdclass=dict(build_ext=CMakeBuild),
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
