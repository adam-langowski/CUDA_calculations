#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>
#include <chrono>
#include <cmath>
#include <tuple>
#include <string>
#include <cmath>
#include <cstdlib>  
#include <ctime>    

// Kernel CUDA do obliczeń
__global__ void calculateOnGPU(const float* input, float* output, int N, int R) {
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

void generateRandomData(std::vector<float>& data) {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (auto& val : data) {
        val = static_cast<float>(std::rand()) / (static_cast<float>(RAND_MAX / 100));
    }
}

void saveParamsToFile(int N, int R, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        exit(1);
    }
    outFile << N << " " << R << std::endl;
    outFile.close();
}

void saveDataToFile(const std::vector<float>& data, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        exit(1);
    }
    outFile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    outFile.close();
}

int main() {
    int R_small = 3;  // R mniejsze od BS
    int R_large = 12; // R większe od BS
    int BS = 8;       // rozmiar bloku

    std::vector<int> N_values = { 32, 64, 128, 256, 512, 1024, 1256, 1512 }; // Testowe wartości N

    // Zbiorcze wyniki
    std::vector<std::tuple<int, int, int, double, double, double>> results;

    for (int R : {R_small, R_large}) {
        for (int N : N_values) {
            int inputSize = N * N;
            int outputSize = (N - 2 * R) * (N - 2 * R);
            int totalOps = outputSize * (2 * R + 1) * (2 * R + 1);

            std::vector<float> input(inputSize);
            std::vector<float> output(outputSize, 0.0f);

            generateRandomData(input);

            float* d_input, * d_output;
            checkCudaError(cudaMalloc((void**)&d_input, inputSize * sizeof(float)), "Failed to allocate device input memory");
            checkCudaError(cudaMalloc((void**)&d_output, outputSize * sizeof(float)), "Failed to allocate device output memory");

            checkCudaError(cudaMemcpy(d_input, input.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy input data to device");

            dim3 threadsPerBlock(BS, BS);
            dim3 numBlocks((N - 2 * R + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (N - 2 * R + threadsPerBlock.y - 1) / threadsPerBlock.y);

            auto start = std::chrono::high_resolution_clock::now();

            calculateOnGPU << <numBlocks, threadsPerBlock >> > (d_input, d_output, N, R);

            checkCudaError(cudaGetLastError(), "Kernel launch failed");
            checkCudaError(cudaDeviceSynchronize(), "Kernel synchronization failed");

            auto end = std::chrono::high_resolution_clock::now();

            checkCudaError(cudaMemcpy(output.data(), d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy output data to host");

            std::chrono::duration<double> duration = end - start;
            double seconds = duration.count();
            double flops = totalOps / seconds;
            double cgma = static_cast<double>(totalOps) / (inputSize * sizeof(float) + outputSize * sizeof(float));

            // Zapis wyników do zbiorczych wyników
            results.push_back(std::make_tuple(N, R, BS, seconds, flops, cgma));

            // Wyświetlenie wyników
            std::cout << "N = " << N << ", R = " << R << ", BS = " << BS << std::endl;
            std::cout << "Czas obliczeń: " << seconds << " seconds" << std::endl;
            std::cout << "Wydajność obliczeń: " << flops << " FLOP/s" << std::endl;
            std::cout << "Arithmetic Intensity (CGMA): " << cgma << " FLOP/byte" << std::endl;
            std::cout << std::endl;

            cudaFree(d_input);
            cudaFree(d_output);
        }
    }

    // Wyświetlenie zbiorczych wyników
    std::cout << "\nZbiorcze wyniki:" << std::endl;
    std::cout << "N\tR\tBS\tCzas(s)\tFLOP/s\tCGMA" << std::endl;
    for (const auto& result : results) {
        std::cout << std::get<0>(result) << "\t" << std::get<1>(result) << "\t" << std::get<2>(result) << "\t"
            << std::get<3>(result) << "\t" << std::get<4>(result) << "\t" << std::get<5>(result) << std::endl;
    }

    return 0;
}
