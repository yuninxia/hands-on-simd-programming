#include "../../include/simd_utils.h"
#include <chrono>
#include <iostream>
#include <iomanip>

/**
 * 01_Basics/02_initializing_data - Different ways to initialize SIMD vectors
 * 
 * This example demonstrates various methods to initialize SIMD vectors:
 * 1. _mm256_setzero_ps/pd/si256 - Initialize all elements to zero
 * 2. _mm256_set1_ps/pd/epi32/etc - Initialize all elements to the same value
 * 3. _mm256_set_ps/pd/epi32/etc - Initialize each element individually
 * 4. _mm256_setr_ps/pd/epi32/etc - Initialize each element in reverse order
 * 
 * We'll also compare the performance of SIMD initialization vs. standard array initialization.
 */

// Constants
constexpr int NUM_ITERATIONS = 1000000;

template <typename T, size_t N>
void printArray(const T (&arr)[N], const std::string &description) {
    std::cout << description << ": ";
    for (size_t i = 0; i < N; ++i) {
        std::cout << arr[i] << ", ";
    }
    std::cout << std::endl;
}

void copyFromSIMD(float* dest, const __m256& src) {
    _mm256_storeu_ps(dest, src);
}

void copyFromSIMD(double* dest, const __m256d& src) {
    _mm256_storeu_pd(dest, src);
}

void copyFromSIMD(int* dest, const __m256i& src) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dest), src);
}

void copyFromSIMD(short* dest, const __m256i& src) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dest), src);
}

int main() {
    std::cout << "=== SIMD Data Initialization Methods ===" << std::endl;
    std::cout << std::endl;

    // --------- 1. Zero Initialization (_mm256_setzero_*) -------------
    std::cout << "1. Zero Initialization (_mm256_setzero_*)" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Initializes all elements of a SIMD vector to zero." << std::endl;
    std::cout << std::endl;

    // Standard method for float array
    float std_float_array[8];
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        for (int lane = 0; lane < 8; ++lane) {
            std_float_array[lane] = 0.0f;
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    // SIMD method for float vector
    __m256 simd_float_vec;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        simd_float_vec = _mm256_setzero_ps();
    }
    stop = std::chrono::high_resolution_clock::now();
    auto duration_simd = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    // Print results
    std::cout << "Float Zero Initialization:" << std::endl;
    std::cout << "  Standard method: " << duration_std.count() << " microseconds" << std::endl;
    std::cout << "  SIMD method:     " << duration_simd.count() << " microseconds" << std::endl;
    std::cout << "  Speedup:         " << std::fixed << std::setprecision(2) 
              << static_cast<double>(duration_std.count()) / duration_simd.count() << "x" << std::endl;
    
    // Print the SIMD vector
    print_m256(simd_float_vec, "Zero-initialized float vector");
    
    // Also demonstrate zero initialization for integers and doubles
    __m256i simd_int_vec = _mm256_setzero_si256();
    __m256d simd_double_vec = _mm256_setzero_pd();
    
    print_m256i(simd_int_vec, "Zero-initialized integer vector");
    print_m256d(simd_double_vec, "Zero-initialized double vector");
    std::cout << std::endl;

    // --------- 2. Broadcast Initialization (_mm256_set1_*) -------------
    std::cout << "2. Broadcast Initialization (_mm256_set1_*)" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Initializes all elements of a SIMD vector to the same value." << std::endl;
    std::cout << std::endl;
    
    // Standard method for double array
    double std_double_array[4];
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        for (int lane = 0; lane < 4; ++lane) {
            std_double_array[lane] = 10.0;
        }
    }
    stop = std::chrono::high_resolution_clock::now();
    duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    // SIMD method for double vector
    __m256d simd_double_vec2;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        simd_double_vec2 = _mm256_set1_pd(10.0);
    }
    stop = std::chrono::high_resolution_clock::now();
    duration_simd = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    // Print results
    std::cout << "Double Broadcast Initialization:" << std::endl;
    std::cout << "  Standard method: " << duration_std.count() << " microseconds" << std::endl;
    std::cout << "  SIMD method:     " << duration_simd.count() << " microseconds" << std::endl;
    std::cout << "  Speedup:         " << std::fixed << std::setprecision(2) 
              << static_cast<double>(duration_std.count()) / duration_simd.count() << "x" << std::endl;
    
    // Print the SIMD vector
    print_m256d(simd_double_vec2, "Broadcast-initialized double vector (10.0)");
    
    // Also demonstrate broadcast initialization for floats and integers
    __m256 simd_float_vec2 = _mm256_set1_ps(42.0f);
    __m256i simd_int_vec2 = _mm256_set1_epi32(100);
    
    print_m256(simd_float_vec2, "Broadcast-initialized float vector (42.0)");
    print_m256i(simd_int_vec2, "Broadcast-initialized integer vector (100)");
    std::cout << std::endl;

    // --------- 3. Individual Element Initialization (_mm256_set_*) -------------
    std::cout << "3. Individual Element Initialization (_mm256_set_*)" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Initializes each element of a SIMD vector individually." << std::endl;
    std::cout << "Note: Elements are specified in reverse order (high to low)." << std::endl;
    std::cout << std::endl;
    
    // Standard method for int array
    int std_int_array[8];
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        for (int lane = 0; lane < 8; ++lane) {
            std_int_array[lane] = lane + 1;
        }
    }
    stop = std::chrono::high_resolution_clock::now();
    duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    // SIMD method for int vector
    __m256i simd_int_vec3;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        // Note: _mm256_set_epi32 takes arguments in reverse order (high to low)
        simd_int_vec3 = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);
    }
    stop = std::chrono::high_resolution_clock::now();
    duration_simd = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    // Print results
    std::cout << "Integer Individual Initialization:" << std::endl;
    std::cout << "  Standard method: " << duration_std.count() << " microseconds" << std::endl;
    std::cout << "  SIMD method:     " << duration_simd.count() << " microseconds" << std::endl;
    std::cout << "  Speedup:         " << std::fixed << std::setprecision(2) 
              << static_cast<double>(duration_std.count()) / duration_simd.count() << "x" << std::endl;
    
    // Print the SIMD vector
    print_m256i(simd_int_vec3, "Individually-initialized integer vector");
    
    // Also demonstrate individual initialization for floats and doubles
    __m256 simd_float_vec3 = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
    __m256d simd_double_vec3 = _mm256_set_pd(4.0, 3.0, 2.0, 1.0);
    
    print_m256(simd_float_vec3, "Individually-initialized float vector");
    print_m256d(simd_double_vec3, "Individually-initialized double vector");
    std::cout << std::endl;

    // --------- 4. Reverse Order Initialization (_mm256_setr_*) -------------
    std::cout << "4. Reverse Order Initialization (_mm256_setr_*)" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Initializes each element of a SIMD vector individually in natural order." << std::endl;
    std::cout << "Note: Elements are specified in natural order (low to high)." << std::endl;
    std::cout << std::endl;
    
    // Standard method for short array
    short std_short_array[16];
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        for (int lane = 0; lane < 16; ++lane) {
            std_short_array[lane] = static_cast<short>(lane + 1);
        }
    }
    stop = std::chrono::high_resolution_clock::now();
    duration_std = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    // SIMD method for short vector
    __m256i simd_short_vec;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        // Note: _mm256_setr_epi16 takes arguments in natural order (low to high)
        simd_short_vec = _mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    }
    stop = std::chrono::high_resolution_clock::now();
    duration_simd = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    // Print results
    std::cout << "Short Reverse Order Initialization:" << std::endl;
    std::cout << "  Standard method: " << duration_std.count() << " microseconds" << std::endl;
    std::cout << "  SIMD method:     " << duration_simd.count() << " microseconds" << std::endl;
    std::cout << "  Speedup:         " << std::fixed << std::setprecision(2) 
              << static_cast<double>(duration_std.count()) / duration_simd.count() << "x" << std::endl;
    
    // Print the SIMD vector (first 8 elements)
    // Note: We need to extract the shorts from the __m256i
    short short_array[16];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(short_array), simd_short_vec);
    
    std::cout << "Reverse-initialized short vector: [";
    for (int i = 0; i < 15; i++) {
        std::cout << short_array[i] << ", ";
    }
    std::cout << short_array[15] << "]" << std::endl;
    
    // Also demonstrate reverse initialization for floats
    __m256 simd_float_vec4 = _mm256_setr_ps(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
    print_m256(simd_float_vec4, "Reverse-initialized float vector");
    
    return 0;
}

