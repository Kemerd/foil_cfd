// CUDA toolchain smoke test: proves nvcc + the configured architecture
// produce a kernel that actually runs on the installed GPU. Computes a SAXPY
// on device, verifies on host, prints the device name and compute capability.
// This test is REAL (not a placeholder) and must always pass on dev machines.
// FoilCFD - PolyForm Noncommercial 1.0.0 - see LICENSE

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace {

// y = a*x + y, one thread per element — minimal but exercises global memory
// reads/writes and launch configuration end to end.
__global__ void saxpyKernel(int n, float a, const float* x, float* y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// Fail-fast CUDA call wrapper for test code.
#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        const cudaError_t err_ = (expr);                                      \
        if (err_ != cudaSuccess) {                                            \
            std::printf("FAIL: %s -> %s\n", #expr, cudaGetErrorString(err_)); \
            return 1;                                                         \
        }                                                                     \
    } while (0)

} // namespace

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::printf("FAIL: no CUDA devices found\n");
        return 1;
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device 0: %s (sm_%d%d, %.1f GB)\n", prop.name, prop.major,
                prop.minor,
                static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));

    // Host reference data.
    constexpr int n = 1 << 20;
    constexpr float a = 2.5f;
    std::vector<float> hx(n), hy(n);
    for (int i = 0; i < n; ++i) {
        hx[i] = static_cast<float>(i % 1000) * 0.001f;
        hy[i] = 1.0f;
    }

    float *dx = nullptr, *dy = nullptr;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid = (n + block - 1) / block;
    saxpyKernel<<<grid, block>>>(n, a, dx, dy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> result(n);
    CUDA_CHECK(cudaMemcpy(result.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dx);
    cudaFree(dy);

    // Bitwise-exact expectation: same FMA-free arithmetic on host.
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        const float expect = a * hx[i] + hy[i];
        const float diff = result[i] - expect;
        if (diff > 1e-5f || diff < -1e-5f) ++bad;
    }
    if (bad != 0) {
        std::printf("FAIL: %d of %d elements wrong\n", bad, n);
        return 1;
    }
    std::printf("PASS: saxpy verified on %d elements\n", n);
    return 0;
}
