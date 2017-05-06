#!/bin/bash

#First open another terminal and run:
#sudo docker run docker.paddlepaddle.org/book:latest

#This is to put docker run on background,but it is useless
#echo -e '\032'

sudo docker run -it -v ~/Skye/ASC17Paddle:/ASC17Paddle docker.paddlepaddle.org/book:latest /bin/bash
export PYTHONHOME=/usr/local
./train.sh

#For exit docker
#exit
