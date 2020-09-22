#!/bin/bash
echo Downloading Dataset

wget -nc http://www.openslr.org/resources/12/train-clean-100.tar.gz

tar -xf train-clean-100.tar.gz

mkdir models