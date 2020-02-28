#include <stdio.h>
#include "gputimer.h"
#include "utils.h"


const int N=1024;  
const int K=16;

// to be launched with one thread per element, in (tilesize)x(tilesize) threadblocks
// thread blocks read & write tiles, in coalesced fashion
// adjacent threads read adjacent input elements, write adjacent output elmts

__global__ void
transpose_parallel_per_element_tiled(float in[], float out[])
{
    
    int in_corner_i  = blockIdx.x * K, in_corner_j  = blockIdx.y * K;
    int out_corner_i = blockIdx.y * K, out_corner_j = blockIdx.x * K;


    int x = threadIdx.x, y = threadIdx.y;


    __shared__ float tile[K][K];


    
    tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y)*N];
    __syncthreads();
    
    out[(out_corner_i + x) + (out_corner_j + y)*N] = tile[x][y];
}