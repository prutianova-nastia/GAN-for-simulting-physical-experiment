#!/bin/sh

image_name=dockeranastasiia/ganproject:1

docker run --runtime nvidia -u $(id -u):$(id -g) --env HOME=`pwd`  -v `pwd`:`pwd` -it $image_name