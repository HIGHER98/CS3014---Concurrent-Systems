#!/bin/bash

gcc -O3 -msse4 conv-harness.c -fopenmp

#Recommended tests
./a.out 16 16 5 1024 128
./a.out 32 32 3 512 256
./a.out 64 64 1 256 64
./a.out 128 128 7 128 128
./a.out 256 256 5 64 64
./a.out 512 512 7 32 32



###All the below tests were ran on a shared 64 core server. I was running a htop simultaneously and could see other activities being ran on the server were causing CPU activity to fluctuate. For this reason these tests are not 100% accurate

#################################

###First round
#Speedup: 17.363607
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 16.919711
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 5.656258
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 32.960659
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 30.806848
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 22.882252
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#################################

###Second round
#Speedup: 18.764950
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 16.338502
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 9.354031
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 33.985529
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 40.238549
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 20.275225
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#################################

###Third round
#Speedup: 14.276586
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 20.033951
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 6.592450
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 30.843405
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 22.841837
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 19.064867
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#################################

###Fourth round
#Speedup: 16.299275
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 17.894892
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 6.767493
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 27.302812
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 51.157856
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Speedup: 25.930428
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#################################

####Round five - with times printed to stdout

#Original conv time: 2801437 microseconds
#Team conv time: 153790 microseconds
#Speedup: 18.215989
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Original conv time: 3561736 microseconds
#Team conv time: 140273 microseconds
#Speedup: 25.391458
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Original conv time: 610052 microseconds
#Team conv time: 83944 microseconds
#Speedup: 7.267369
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Original conv time: 32758142 microseconds
#Team conv time: 979380 microseconds
#Speedup: 33.447836
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Original conv time: 18615240 microseconds
#Team conv time: 606817 microseconds
#Speedup: 30.676860
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Original conv time: 32429742 microseconds
#Team conv time: 1608335 microseconds
#Speedup: 20.163549
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#################################

####Round six
#Original conv time: 2678887 microseconds
#Team conv time: 159533 microseconds
#Speedup: 16.792056
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Original conv time: 3924156 microseconds
#Team conv time: 156077 microseconds
#Speedup: 25.142436
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Original conv time: 742529 microseconds
#Team conv time: 103819 microseconds
#Speedup: 7.152149
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Original conv time: 32757782 microseconds
#Team conv time: 891821 microseconds
#Speedup: 36.731342
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Original conv time: 19083170 microseconds
#Team conv time: 584631 microseconds
#Speedup: 32.641393
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Original conv time: 32641885 microseconds
#Team conv time: 1592654 microseconds
#Speedup: 20.495277
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#################################

###Round seven

#Original conv time: 2448988 microseconds
#Team conv time: 155791 microseconds
#Speedup: 15.719701
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Original conv time: 3867124 microseconds
#Team conv time: 214249 microseconds
#Speedup: 18.049671
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Original conv time: 688012 microseconds
#Team conv time: 108724 microseconds
#Speedup: 6.328060
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Original conv time: 32726872 microseconds
#Team conv time: 925845 microseconds
#Speedup: 35.348111
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Original conv time: 18957267 microseconds
#Team conv time: 561699 microseconds
#Speedup: 33.749868
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

#Original conv time: 32599975 microseconds
#Team conv time: 1605737 microseconds
#Speedup: 20.302188
#COMMENT: sum of absolute differences (0.000000)  within acceptable range (0.062500)

