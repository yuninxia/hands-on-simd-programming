#include "../../include/simd_utils.h"
#include <iostream>
#include <iomanip>

/**
 * 01_Basics/03_binding_with_unions - Techniques for accessing SIMD data
 * 
 * This example demonstrates different ways to access and manipulate data in SIMD vectors:
 * 1. Using pointer conversion (reinterpret_cast)
 * 2. Using unions to create an alias between SIMD types and arrays
 * 3. Using the _mm256_store_* and _mm256_load_* functions
 * 4. Using the extract and insert element functions
 * 
 * Each method has its advantages and use cases.
 */

int main() {
    std::cout << "=== Accessing SIMD Data ===" << std::endl;
    std::cout << std::endl;

    // --------- 1. Pointer Conversion -------------
    std::cout << "1. Pointer Conversion" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Using reinterpret_cast to convert between SIMD types and arrays." << std::endl;
    std::cout << "This is a simple but potentially unsafe method." << std::endl;
    std::cout << std::endl;
    
    // Initialize a SIMD vector with ascending values
    __m256 simd_vec1 = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
    
    // Access the data using pointer conversion
    float* float_ptr = reinterpret_cast<float*>(&simd_vec1);
    
    // Print the data
    std::cout << "SIMD vector values via pointer: [";
    for (int i = 0; i < 7; i++) {
        std::cout << float_ptr[i] << ", ";
    }
    std::cout << float_ptr[7] << "]" << std::endl;
    
    // Modify the data through the pointer
    std::cout << "Modifying values via pointer..." << std::endl;
    float_ptr[0] = 100.0f;
    float_ptr[4] = 200.0f;
    
    // Print the modified SIMD vector
    print_m256(simd_vec1, "Modified SIMD vector");
    std::cout << std::endl;

    // --------- 2. Using Unions -------------
    std::cout << "2. Using Unions" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Using unions to create an alias between SIMD types and arrays." << std::endl;
    std::cout << "This is a cleaner and safer approach than pointer conversion." << std::endl;
    std::cout << std::endl;
    
    // Define a union for float SIMD vector
    union FloatSIMD {
        __m256 v;
        float a[8];
    };
    
    // Initialize the union with a SIMD vector
    FloatSIMD float_union;
    float_union.v = _mm256_set_ps(16.0f, 14.0f, 12.0f, 10.0f, 8.0f, 6.0f, 4.0f, 2.0f);
    
    // Access the data through the array
    std::cout << "SIMD vector values via union: [";
    for (int i = 0; i < 7; i++) {
        std::cout << float_union.a[i] << ", ";
    }
    std::cout << float_union.a[7] << "]" << std::endl;
    
    // Modify the data through the array
    std::cout << "Modifying values via union..." << std::endl;
    float_union.a[1] = 42.0f;
    float_union.a[6] = 99.0f;
    
    // Print the modified SIMD vector
    print_m256(float_union.v, "Modified SIMD vector (union)");
    
    // Using our utility union from simd_utils.h
    float8 float8_union;
    float8_union.v = _mm256_set1_ps(5.0f);
    float8_union.a[2] = 10.0f;
    float8_union.a[5] = 20.0f;
    
    print_m256(float8_union.v, "Using float8 union from simd_utils.h");
    std::cout << std::endl;

    // --------- 3. Store and Load Functions -------------
    std::cout << "3. Store and Load Functions" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Using _mm256_store_* and _mm256_load_* functions to transfer data." << std::endl;
    std::cout << "This is the recommended approach for most situations." << std::endl;
    std::cout << std::endl;
    
    // Initialize a SIMD vector
    __m256 simd_vec3 = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
    
    // Allocate aligned memory for the array
    float* aligned_array = aligned_alloc<float>(8);
    
    // Store the SIMD vector to the array
    _mm256_store_ps(aligned_array, simd_vec3);
    
    // Print the array
    std::cout << "SIMD vector values via store: [";
    for (int i = 0; i < 7; i++) {
        std::cout << aligned_array[i] << ", ";
    }
    std::cout << aligned_array[7] << "]" << std::endl;
    
    // Modify the array
    std::cout << "Modifying values in the array..." << std::endl;
    aligned_array[3] = 30.0f;
    aligned_array[7] = 80.0f;
    
    // Load the modified array back to a SIMD vector
    __m256 modified_vec = _mm256_load_ps(aligned_array);
    
    // Print the modified SIMD vector
    print_m256(modified_vec, "Modified SIMD vector (store/load)");
    std::cout << std::endl;

    // --------- 4. Extract and Insert Elements -------------
    std::cout << "4. Extract and Insert Elements" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Using _mm256_extract_* and _mm256_insert_* functions to access individual elements." << std::endl;
    std::cout << "This is useful when you only need to access a few elements." << std::endl;
    std::cout << std::endl;
    
    // Initialize a SIMD vector with integers
    __m256i simd_int_vec = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);
    
    // Extract individual elements
    // Note: For AVX2, we need to extract 128-bit lanes first, then extract from those
    __m128i low_lane = _mm256_extracti128_si256(simd_int_vec, 0);  // Extract lower 128 bits
    __m128i high_lane = _mm256_extracti128_si256(simd_int_vec, 1); // Extract upper 128 bits
    
    int element0 = _mm_extract_epi32(low_lane, 0);  // Extract element 0
    int element3 = _mm_extract_epi32(low_lane, 3);  // Extract element 3
    int element4 = _mm_extract_epi32(high_lane, 0); // Extract element 4
    int element7 = _mm_extract_epi32(high_lane, 3); // Extract element 7
    
    std::cout << "Extracted elements: " << element0 << ", " << element3 << ", " 
              << element4 << ", " << element7 << std::endl;
    
    // Insert elements
    // For inserting, we need to create new 128-bit vectors and then combine them
    __m128i new_low = _mm_insert_epi32(low_lane, 100, 1);  // Replace element 1
    __m128i new_high = _mm_insert_epi32(high_lane, 200, 2); // Replace element 6
    
    // Combine the lanes back into a 256-bit vector
    __m256i modified_int_vec = _mm256_setr_m128i(new_low, new_high);
    
    // Print the modified vector
    print_m256i(modified_int_vec, "Modified integer vector (extract/insert)");
    
    // Clean up
    free(aligned_array);
    
    return 0;
}
