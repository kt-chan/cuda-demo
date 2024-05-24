#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

#define Debug false

using namespace std;

void rand(int *a, int n)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % n;
        if (Debug)
            cout << a[i] << "\t";
    }
    if (Debug)
        cout << "\n";
}

void cpucal(int *a, int *b, int *c, int n)
{
    // calculate the average of each of a / each of b
    int sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum = 0;
        for (int j = 0; j < n; j++)
        {
            if (b[j] != 0)
            {
                int delta = a[i] / b[j];
                if (j == 0)
                    sum = delta;
                else
                    sum += delta / j;
            }
        }
        c[i] = sum;
    }
}

__global__ void gpucal_kernel(int *a, int *b, int *c, int n)
{
    // Calculate the global thread index
    int globalThreadId = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;

    // Calculate 2D indices from the global thread index
    int i = globalThreadId / n;
    int j = globalThreadId % n;

    // Ensure both i and j are within bounds before accessing a and b
    if (i < n && j < n && b[j] != 0)
    {
        int valueToAdd = c[i] + (j == 0 ? a[i] / b[0] : (a[i] / b[j]) / j);
        atomicAdd(&c[i], valueToAdd);
    }
}

void gpucal (int *a, int *b, int *c, int n ) {
  int *d_a , *d_b , *d_c;
  cudaMalloc(&d_a, n*sizeof(int));
  cudaMalloc(&d_b, n*sizeof(int));
  cudaMalloc(&d_c, n*sizeof(int));
  cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice);

  // define kernel call
  int grids = max(1, (n+255)/256);
  gpucal_kernel<<<grids,256>>>(d_a, d_b, d_c, n);

  // Sync Device to Host
  cudaDeviceSynchronize();

  cudaMemcpy(c, d_c, n*sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

bool checkGPU()
{
    const char *gpu_env = getenv("COLAB_GPU");
    if (gpu_env && atoi(gpu_env) > 0)
    {
        cout << "A GPU is connected." << endl;
        return true;
    }
    else
    {
        cout << "No accelerator is connected." << endl;
        return false;
    }
}

int main(void)
{

    bool GPU = checkGPU();

    int N = 1 << 15;
    cout << "n:\t" << N << "\n\n";

    int *a, *b, *c;
    if (!GPU)
    {
        a = (int *)malloc(N * sizeof(int));
        b = (int *)malloc(N * sizeof(int));
        c = (int *)malloc(N * sizeof(int));
    }
    else
    {
        cudaMallocHost(&a, N * sizeof(int));
        cudaMallocHost(&b, N * sizeof(int));
        cudaMallocHost(&c, N * sizeof(int));
    }
    if (Debug)
        cout << "a array list:\t";
    rand(a, N);
    if (Debug)
        cout << "b array list:\t";
    rand(b, N);

    cout << "\n\n@CPU, summing value of size(n) * size(n) ... \n";

    clock_t t;

    // calling cpu
    t = clock(); // start time
    cpucal(a, b, c, N);
    t = clock() - t; // total time = end time - start time

    int vmax = 0;
    for (int i = 0; i < N; i++)
    {
        vmax = (vmax > c[i]) ? vmax : c[i];
    }
    cout << vmax << "\n";

    printf("CPU Avg time = %lf ms.\n", ((((float)t) / CLOCKS_PER_SEC) * 1000));

    cout << "\n\n@GPU, summing value of size(n) * size(n) ... \n";

    // calling cpu
    t = clock(); // start time
    gpucal(a, b, c, N);
    t = clock() - t; // total time = end time - start time

    vmax = 0;
    for (int i = 0; i < N; i++)
    {
        vmax = (vmax > c[i]) ? vmax : c[i];
    }
    cout << vmax << "\n";

    printf("GPU Avg time = %lf ms.\n", ((((float)t) / CLOCKS_PER_SEC) * 1000));
    if (!GPU)
    {
        free(a);
        free(b);
        free(c);
    }
    else
    {
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
    }

    return 0;
}