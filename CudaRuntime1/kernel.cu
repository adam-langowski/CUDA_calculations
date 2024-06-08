#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>
#include <chrono>
#include <cmath>

// Kernel CUDA do obliczeń
__global__ void calculateOnGPU(const float* input, float* output, int N, int R) {
    extern __shared__ float sharedMem[];
    int outSize = N - 2 * R;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < outSize && j < outSize) {
        float sum = 0.0f;
        for (int x = -R; x <= R; ++x) {
            for (int y = -R; y <= R; ++y) {
                sum += input[(i + R + x) * N + (j + R + y)];
            }
        }
        output[i * outSize + j] = sum;
    }
}

void checkCudaError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        std::cerr << msg << " Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void loadDataFromFile(std::vector<float>& data, const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Cannot open file for reading: " << filename << std::endl;
        exit(1);
    }
    inFile.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    inFile.close();
}

bool compareResults(const std::vector<float>& cpuResults, const std::vector<float>& gpuResults, float tolerance = 1e-5) {
    if (cpuResults.size() != gpuResults.size()) {
        return false;
    }
    for (size_t i = 0; i < cpuResults.size(); ++i) {
        if (std::fabs(cpuResults[i] - gpuResults[i]) > tolerance) {
            std::cout << "Difference at index " << i << ": CPU = " << cpuResults[i] << ", GPU = " << gpuResults[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    int N = 32;
    int R = 1;
    int inputSize = N * N;
    int outputSize = (N - 2 * R) * (N - 2 * R);
    int totalOps = outputSize * (2 * R + 1) * (2 * R + 1);

    std::vector<float> input(inputSize); // Inicjalizacja losowymi danymi
    std::vector<float> output(outputSize, 0.0f);
    std::vector<float> cpuOutput(outputSize, 0.0f);

    // Wczytanie danych wejściowych z pliku
    loadDataFromFile(input, "input_data.bin");
    loadDataFromFile(cpuOutput, "cpu_output.bin");

    // Wyświetlenie tablicy wejściowej
    std::cout << "Input Array:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << input[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    float* d_input, * d_output;
    checkCudaError(cudaMalloc((void**)&d_input, inputSize * sizeof(float)), "Failed to allocate device input memory");
    checkCudaError(cudaMalloc((void**)&d_output, outputSize * sizeof(float)), "Failed to allocate device output memory");

    checkCudaError(cudaMemcpy(d_input, input.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy input data to device");

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N - 2 * R + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N - 2 * R + threadsPerBlock.y - 1) / threadsPerBlock.y);
    size_t sharedMemSize = (threadsPerBlock.x + 2 * R) * (threadsPerBlock.y + 2 * R) * sizeof(float);

    auto start = std::chrono::high_resolution_clock::now();

    calculateOnGPU << <numBlocks, threadsPerBlock, sharedMemSize >> > (d_input, d_output, N, R);

    auto end = std::chrono::high_resolution_clock::now();

    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    checkCudaError(cudaDeviceSynchronize(), "Kernel synchronization failed");

    checkCudaError(cudaMemcpy(output.data(), d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy output data to host");

    std::chrono::duration<double> duration = end - start;
    double seconds = duration.count();
    double flops = totalOps / seconds;
    double cgma = static_cast<double>(totalOps) / (inputSize * sizeof(float) + outputSize * sizeof(float));

    // Wyświetlenie wyników
    std::cout << "Output Array (GPU):" << std::endl;
    for (int i = 0; i < outputSize; ++i) {
        if (i % (N - 2 * R) == 0) std::cout << std::endl;
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Czas obliczen: " << seconds << " seconds" << std::endl;
    std::cout << "Wydajność obliczen : " << flops << " FLOP/s" << std::endl;
    std::cout << "Arithmetic Intensity (CGMA): " << cgma << " FLOP/byte" << std::endl;
    std::cout << "Rozmiar pamięci współdzielonej przez blok watkow: " << sharedMemSize << " bytes" << std::endl;

    // Porównanie wyników
    if (compareResults(cpuOutput, output)) {
        std::cout << "Wyniki obliczen są poprawne!" << std::endl;
    }
    else {
        std::cout << "Wyniki obliczen są niepoprawne!" << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
