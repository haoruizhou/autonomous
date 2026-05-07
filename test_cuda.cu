#include <stdio.h>
__global__ void hello() {
    printf("Hello from block %d, thread %d\\n", blockIdx.x, threadIdx.x);
}
int main() {
    hello<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
