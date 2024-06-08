#include "pch.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <random>

// Funkcja obliczająca tablicę wyjściową na CPU
void calculateOnCPU(const std::vector<float>& input, std::vector<float>& output, int N, int R) {
    int outSize = N - 2 * R;
    for (int i = 0; i < outSize; ++i) {
        for (int j = 0; j < outSize; ++j) {
            float sum = 0.0f;
            for (int x = -R; x <= R; ++x) {
                for (int y = -R; y <= R; ++y) {
                    sum += input[(i + R + x) * N + (j + R + y)];
                }
            }
            output[i * outSize + j] = sum;
        }
    }
}

// Funkcja generująca losowe dane
void generateRandomData(std::vector<float>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 100);

    for (auto& val : data) {
        val = dis(gen);
    }
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

void saveParamsToFile(int N, int R, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        exit(1);
    }
    outFile << N << " " << R << std::endl;
    outFile.close();
}

struct Result {
    int N;
    int R;
    double seconds;
    double flops;
    double cgma;
};

int main() {
    std::vector<int> N_values = { 32, 64, 128, 256 }; // Testowe wartości N
    std::vector<int> R_values = { 3, 6, 9, 12 };    // Testowe wartości R
    std::vector<Result> results; // Zbiorcze wyniki

    for (int N : N_values) {
        for (int R : R_values) {
            int inputSize = N * N;
            int outputSize = (N - 2 * R) * (N - 2 * R);
            int totalOps = outputSize * (2 * R + 1) * (2 * R + 1);

            std::vector<float> input(inputSize); // Inicjalizacja losowymi danymi
            std::vector<float> output(outputSize, 0.0f);

            generateRandomData(input);

            // Zapisanie danych wejściowych do pliku
            saveDataToFile(input, "input_data.bin");

            // Zapisanie parametrów do pliku
            saveParamsToFile(N, R, "params.txt");

            // Wyświetlenie tablicy wejściowej
            std::cout << "Input Array (N = " << N << ", R = " << R << "):" << std::endl;
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    std::cout << input[i * N + j] << " ";
                }
                std::cout << std::endl;
            }

            auto start = std::chrono::high_resolution_clock::now();
            calculateOnCPU(input, output, N, R);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> duration = end - start;
            double seconds = duration.count();
            double flops = totalOps / seconds;
            double cgma = static_cast<double>(totalOps) / (inputSize * sizeof(float) + outputSize * sizeof(float));

            // Zapisanie wyników CPU do pliku
            saveDataToFile(output, "cpu_output.bin");

            // Wyświetlenie wyników
            //std::cout << "Output Array (CPU, N = " << N << ", R = " << R << "):" << std::endl;
            //for (int i = 0; i < outputSize; ++i) {
            //    if (i % (N - 2 * R) == 0) std::cout << std::endl;
            //     std::cout << output[i] << " ";
            // }
            //std::cout << std::endl;

            std::cout << "Czas obliczeń: " << seconds << " seconds" << std::endl;
            std::cout << "Wydajność obliczeń: " << flops << " FLOP/s" << std::endl;
            std::cout << "Arithmetic Intensity (CGMA): " << cgma << " FLOP/byte" << std::endl;

            // Zapisanie wyników do zbiorczej struktury danych
            results.push_back({ N, R, seconds, flops, cgma });
        }
    }

    // Wyświetlenie wyników zbiorczych
    std::cout << "\nZbiorcze wyniki:" << std::endl;
    std::cout << "N\tR\tCzas(s)\tFLOP/s\tCGMA" << std::endl;
    for (const auto& result : results) {
        std::cout << result.N << "\t" << result.R << "\t" << result.seconds << "\t" << result.flops << "\t" << result.cgma << std::endl;
    }

    return 0;
}
