FROM ubuntu:devel

LABEL maintainer = "Jingpeng Wu" \
        email = "jingpeng.wu@gmail.com"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/anaconda3/bin:$PATH

ENV HOME /root
RUN mkdir -p $HOME/workspace/reneu
WORKDIR $HOME/workspace/reneu
COPY . .


WORKDIR $HOME
RUN apt-get update --fix-missing \
    && apt-get install -y -qq --no-install-recommends \
        apt-utils \
        git \
        wget \
        build-essential \
        bzip2 \
        ca-certificates \
    # install miniconda3
    && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/anaconda3 \
    && rm Miniconda3-latest-Linux-x86_64.sh \
    && chmod +w -R /opt/anaconda3 \
    && ln -s /opt/anaconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc 
    # && conda update -n base -c defaults conda 
    # numpy is required for setup.py file
    # && conda install -y numpy

#ENV CPLUS_INCLUDE_PATH "/opt/anaconda3/include":$CPLUS_INCLUDE_PATH
#ENV CPLUS_INCLUDE_PATH "/opt/anaconda3/include/python3.7m":$CPLUS_INCLUDE_PATH
ENV CPLUS_INCLUDE_PATH "/opt/anaconda3/envs/reneu/include":$CPLUS_INCLUDE_PATH
ENV CPLUS_INCLUDE_PATH "/opt/anaconda3/envs/reneu/include/python3.7m":$CPLUS_INCLUDE_PATH
ENV CPLUS_INCLUDE_PATH "$HOME/workspace/reneu/include":$CPLUS_INCLUDE_PATH

WORKDIR $HOME/workspace/reneu
RUN exec bash \
    && conda env create -f environment.yml \
    && conda init bash \
    && conda activate reneu \
    && python setup.py install \
    && pytest -s tests