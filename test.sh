#!/bin/bash

gcc -O3 -msse4 conv-harness.c -fopenmp

#Recommended tests
./a.out 16 16 5 1024 128
./a.out 32 32 3 512 256
./a.out 64 64 1 256 64
./a.out 128 128 7 128 128
./a.out 256 256 5 64 64
./a.out 512 512 7 32 32



###All the below tests were ran on a shared 64 core server. I was running a htop simultaneously and could see other activities being ran on the server were causing CPU activity to fluctuate. For this reason these results are not 100% accurate



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
