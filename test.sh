#!/bin/bash

gcc -O3 -msse4 conv-harness.c -fopenmp


#Gregg's conv time: 739 microseconds
#Team conv time: 494 microseconds
#./a.out 512 512 1 128 32

#./a.out 16 16 3 32 32
#./a.out 16 16 5 32 32
#./a.out 16 16 7 32 32


#Gregg's conv time: 1581795 microseconds
#Team conv time: 1000543 microseconds
./a.out 128 128 3 64 64


#Gregg's conv time: 101978515 microseconds
#Team conv time: 64938434 microseconds
#./a.out 512 512 1 128 1024


#./a.out 512 512 3 128 1024
#./a.out 512 512 3 128 1024
#./a.out 512 512 3 128 1024
#./a.out 512 512 3 128 1024
#./a.out 512 512 3 128 1024

#Gregg's conv time: 100597345 microseconds
#Team conv time: 65248660 microseconds
#./a.out 512 512 1 256 512

#./a.out 512 512 3 256 512
#./a.out 512 512 3 256 512
#./a.out 512 512 3 256 512
#./a.out 512 512 3 256 512
#./a.out 512 512 3 256 512

#Gregg's conv time: 100286075 microseconds
#Team conv time: 63584369 microseconds
#./a.out 512 512 1 512 256

#./a.out 512 512 3 512 256
#./a.out 512 512 3 512 256
#./a.out 512 512 3 512 256
#./a.out 512 512 3 512 256
#./a.out 512 512 3 512 256

#Gregg's conv time: 101653345 microseconds
#Team conv time: 66276738 microseconds
#./a.out 512 512 1 1024 128


#./a.out 512 512 3 1024 128
#./a.out 512 512 3 1024 128
#./a.out 512 512 3 1024 128
#./a.out 512 512 3 1024 128
#./a.out 512 512 3 1024 128
#./a.out 512 512 3 1024 128

#./a.out 512 512 1 2048 32
#./a.out 512 512 3 2048 32
#./a.out 512 512 3 2048 32
#./a.out 512 512 3 2048 32
#./a.out 512 512 3 2048 32
#./a.out 512 512 3 2048 32
