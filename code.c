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
    // (i,j) locations of the tile corners for input & output matrices:
    int in_corner_i  = blockIdx.x * K, in_corner_j  = blockIdx.y * K;
    int out_corner_i = blockIdx.y * K, out_corner_j = blockIdx.x * K;


    int x = threadIdx.x, y = threadIdx.y;


    __shared__ float tile[K][K];


    // coalesced read from global mem, TRANSPOSED write into shared mem:
    tile[y][x] = in[(in_corner_i + x) + (in_corner_j + y)*N];
    __syncthreads();
    // read from shared mem, coalesced write to global mem:
    out[(out_corner_i + x) + (out_corner_j + y)*N] = tile[x][y];
}

void
transpose_CPU(float in[], float out[])
{
    for(int j=0; j < N; j++)
        for(int i=0; i < N; i++)
              out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

__global__ void
transpose_serial(float in[], float out[])
{
    for(int j=0; j < N; j++)
        for(int i=0; i < N; i++)
            out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

__global__ void
transpose_parallel_per_row(float in[], float out[])
{
    int i = threadIdx.x;


    for(int j=0; j < N; j++)
        out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

int main(int argc, char **argv)
{
    int numbytes = N * N * sizeof(float);


    float *in = (float *) malloc(numbytes);
    float *out = (float *) malloc(numbytes);
    float *gold = (float *) malloc(numbytes);


    fill_matrix(in, N);
    transpose_CPU(in, gold);


    float *d_in, *d_out;


    cudaMalloc(&d_in, numbytes);
    cudaMalloc(&d_out, numbytes);
    cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);


    GpuTimer timer;
	
    dim3 blocks(64,64);    //TODO, you need to set the proper blocks per grid
    dim3 threads(16,16);    //TODO, you need to set the proper threads per block


    timer.Start();
    transpose_parallel_per_element_tiled<<<blocks,threads>>>(d_in, d_out);
    timer.Stop();
    cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
    printf("transpose_parallel_per_element_tiled %dx%d: %g ms.\nVerifying ...%s\n",
           K, K, timer.Elapsed(), compare_matrices(out, gold, N) ? "Failed" : "Success");


    cudaFree(d_in);
    cudaFree(d_out);

	

}
