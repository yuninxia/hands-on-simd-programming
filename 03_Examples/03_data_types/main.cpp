#include "../../include/simd_utils.h"
#include <iostream>
#include <iomanip>

/**
 * This example demonstrates SIMD operations with different data types.
 * 
 * We'll explore:
 * 1. Working with different numeric types (float, double, int, short)
 * 2. Converting between different SIMD types
 * 3. Handling different vector widths
 * 4. Performing operations specific to certain data types
 */

int main() {
    std::cout << "=== SIMD Operations with Different Data Types ===" << std::endl;
    std::cout << std::endl;

    // -------- 1. Float operations (32-bit) --------
    std::cout << "1. Float Operations (32-bit, 8 elements per vector)" << std::endl;
    
    // Initialize float vector
    __m256 float_vec1 = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
    __m256 float_vec2 = _mm256_set1_ps(2.0f);  // Set all elements to 2.0
    
    // Perform operations
    __m256 float_sum = _mm256_add_ps(float_vec1, float_vec2);
    __m256 float_product = _mm256_mul_ps(float_vec1, float_vec2);
    
    // Print results
    print_m256(float_vec1, "Float Vector 1");
    print_m256(float_vec2, "Float Vector 2");
    print_m256(float_sum, "Sum (float_vec1 + float_vec2)");
    print_m256(float_product, "Product (float_vec1 * float_vec2)");
    std::cout << std::endl;

    // -------- 2. Double operations (64-bit) --------
    std::cout << "2. Double Operations (64-bit, 4 elements per vector)" << std::endl;
    
    // Initialize double vector
    __m256d double_vec1 = _mm256_set_pd(4.0, 3.0, 2.0, 1.0);
    __m256d double_vec2 = _mm256_set1_pd(3.0);  // Set all elements to 3.0
    
    // Perform operations
    __m256d double_sum = _mm256_add_pd(double_vec1, double_vec2);
    __m256d double_product = _mm256_mul_pd(double_vec1, double_vec2);
    
    // Print results
    print_m256d(double_vec1, "Double Vector 1");
    print_m256d(double_vec2, "Double Vector 2");
    print_m256d(double_sum, "Sum (double_vec1 + double_vec2)");
    print_m256d(double_product, "Product (double_vec1 * double_vec2)");
    std::cout << std::endl;

    // -------- 3. Integer operations (32-bit) --------
    std::cout << "3. Integer Operations (32-bit, 8 elements per vector)" << std::endl;
    
    // Initialize integer vector
    __m256i int_vec1 = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);
    __m256i int_vec2 = _mm256_set1_epi32(10);  // Set all elements to 10
    
    // Perform operations
    __m256i int_sum = _mm256_add_epi32(int_vec1, int_vec2);
    __m256i int_sub = _mm256_sub_epi32(int_vec1, int_vec2);
    
    // Print results
    print_m256i(int_vec1, "Int Vector 1");
    print_m256i(int_vec2, "Int Vector 2");
    print_m256i(int_sum, "Sum (int_vec1 + int_vec2)");
    print_m256i(int_sub, "Difference (int_vec1 - int_vec2)");
    std::cout << std::endl;

    // -------- 4. Type Conversions --------
    std::cout << "4. Type Conversions" << std::endl;
    
    // Convert float to integer (truncation)
    __m256i float_to_int = _mm256_cvttps_epi32(float_vec1);
    print_m256i(float_to_int, "Float to Int (truncated)");
    
    // Convert integer to float
    __m256 int_to_float = _mm256_cvtepi32_ps(int_vec1);
    print_m256(int_to_float, "Int to Float");
    
    // Convert between float and double (need to split/combine)
    // Extract lower 4 floats and convert to double
    __m128 float_low = _mm256_extractf128_ps(float_vec1, 0);
    __m256d float_to_double_low = _mm256_cvtps_pd(float_low);
    print_m256d(float_to_double_low, "Lower 4 Floats to Double");
    
    // Extract upper 4 floats and convert to double
    __m128 float_high = _mm256_extractf128_ps(float_vec1, 1);
    __m256d float_to_double_high = _mm256_cvtps_pd(float_high);
    print_m256d(float_to_double_high, "Upper 4 Floats to Double");
    std::cout << std::endl;

    // -------- 5. Bitwise Operations --------
    std::cout << "5. Bitwise Operations" << std::endl;
    
    // Create test vectors
    __m256i bits1 = _mm256_set1_epi32(0x0F0F0F0F);  // 00001111 00001111 00001111 00001111
    __m256i bits2 = _mm256_set1_epi32(0x33333333);  // 00110011 00110011 00110011 00110011
    
    // Perform bitwise operations
    __m256i bit_and = _mm256_and_si256(bits1, bits2);
    __m256i bit_or = _mm256_or_si256(bits1, bits2);
    __m256i bit_xor = _mm256_xor_si256(bits1, bits2);
    
    // Print results in hex format
    std::cout << "Bits1 (hex): 0x" << std::hex << std::setfill('0') << std::setw(8) 
              << reinterpret_cast<int*>(&bits1)[0] << std::endl;
    std::cout << "Bits2 (hex): 0x" << std::hex << std::setfill('0') << std::setw(8) 
              << reinterpret_cast<int*>(&bits2)[0] << std::endl;
    std::cout << "AND (hex): 0x" << std::hex << std::setfill('0') << std::setw(8) 
              << reinterpret_cast<int*>(&bit_and)[0] << std::endl;
    std::cout << "OR (hex): 0x" << std::hex << std::setfill('0') << std::setw(8) 
              << reinterpret_cast<int*>(&bit_or)[0] << std::endl;
    std::cout << "XOR (hex): 0x" << std::hex << std::setfill('0') << std::setw(8) 
              << reinterpret_cast<int*>(&bit_xor)[0] << std::endl;
    std::cout << std::dec << std::endl;  // Reset to decimal

    // -------- 6. Specialized Operations --------
    std::cout << "6. Specialized Operations" << std::endl;
    
    // Horizontal addition (add adjacent pairs)
    __m256 hadd_result = _mm256_hadd_ps(float_vec1, float_vec2);
    print_m256(hadd_result, "Horizontal Add (pairs from float_vec1, float_vec2)");
    
    // Permute (rearrange elements)
    __m256 permute_result = _mm256_permute_ps(float_vec1, 0b10010011);
    print_m256(permute_result, "Permuted float_vec1");
    
    // Blend (select elements from two vectors based on mask)
    __m256 blend_result = _mm256_blend_ps(float_vec1, float_vec2, 0b10101010);
    print_m256(blend_result, "Blend of float_vec1 and float_vec2");
    
    return 0;
} 