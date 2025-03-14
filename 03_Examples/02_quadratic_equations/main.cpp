#include "../../include/simd_utils.h"
#include <iostream>
#include <math.h>

/**
 * This example demonstrates solving multiple quadratic equations in parallel using SIMD.
 * 
 * For each quadratic equation ax² + bx + c = 0, we compute the discriminant b² - 4ac
 * and then calculate the solution using the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a
 * 
 * We'll solve 8 different quadratic equations simultaneously using AVX2 instructions.
 */

int main() {
	std::cout << "=== Solving Quadratic Equations with SIMD ===" << std::endl;
	std::cout << "This example solves 8 quadratic equations in parallel." << std::endl;
	std::cout << "For each equation ax² + bx + c = 0, we find the smaller root." << std::endl;
	std::cout << std::endl;

	// Allocate aligned memory for coefficients
	float* a = aligned_alloc<float>(8);
	float* b = aligned_alloc<float>(8);
	float* c = aligned_alloc<float>(8);

	// Initialize coefficients for 8 different quadratic equations
	// Equation 1: 5x² + 3x - 1 = 0
	a[0] = 5.0f;  b[0] = 3.0f;  c[0] = -1.0f;
	
	// Equation 2: 12x² + 1x - 5 = 0
	a[1] = 12.0f; b[1] = 1.0f;  c[1] = -5.0f;
	
	// Equation 3: 6x² + 4x - 6 = 0
	a[2] = 6.0f;  b[2] = 4.0f;  c[2] = -6.0f;
	
	// Equation 4: 7x² - 2x - 6 = 0
	a[3] = 7.0f;  b[3] = -2.0f; c[3] = -6.0f;
	
	// Equation 5: 1x² + 2x + 5 = 0 (complex roots, will return NaN)
	a[4] = 1.0f;  b[4] = 2.0f;  c[4] = 5.0f;
	
	// Equation 6: 1x² + 1x + 30 = 0 (complex roots, will return NaN)
	a[5] = 1.0f;  b[5] = 1.0f;  c[5] = 30.0f;
	
	// Equation 7: 1x² + 1x + 35 = 0 (complex roots, will return NaN)
	a[6] = 1.0f;  b[6] = 1.0f;  c[6] = 35.0f;
	
	// Equation 8: 1x² + 1x - 40 = 0
	a[7] = 1.0f;  b[7] = 1.0f;  c[7] = -40.0f;

	// Print the equations we're solving
	std::cout << "Equations to solve:" << std::endl;
	for (int i = 0; i < 8; i++) {
		std::cout << "Equation " << (i+1) << ": " 
				  << a[i] << "x² + " << b[i] << "x + " << c[i] << " = 0" << std::endl;
	}
	std::cout << std::endl;

	// Union to access SIMD results
	float8 result;

	// -------- Standard scalar approach ---------------
	std::cout << "----------- Standard scalar approach -----------" << std::endl;
	
	// Initialize result array with placeholder values
	for (int lane = 0; lane < 8; ++lane) {
		result.a[lane] = std::numeric_limits<float>::quiet_NaN();
	}
	
	// Define the scalar implementation as a lambda for benchmarking
	auto scalar_func = [&]() {
		for (int lane = 0; lane < 8; ++lane) {
			float discriminant = b[lane] * b[lane] - 4.0f * a[lane] * c[lane];
			if (discriminant >= 0) {
				// Calculate the smaller root: (-b - sqrt(discriminant)) / (2*a)
				result.a[lane] = (-b[lane] - sqrtf(discriminant)) / (2.0f * a[lane]);
			} else {
				// Complex roots, set to NaN
				result.a[lane] = std::numeric_limits<float>::quiet_NaN();
			}
		}
	};
	
	// Run the scalar implementation once to get the results
	scalar_func();

	// Print scalar results
	std::cout << "Scalar solutions (smaller root):" << std::endl;
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << "Equation " << (lane+1) << ": ";
		if (std::isnan(result.a[lane])) {
			std::cout << "Complex roots" << std::endl;
		} else {
			std::cout << result.a[lane] << std::endl;
		}
	}
	std::cout << std::endl;

	// -------- SIMD approach ---------------
	std::cout << "----------- SIMD approach -----------" << std::endl;
	
	// Define the SIMD implementation as a lambda for benchmarking
	auto simd_func = [&]() {
		// Load coefficients into SIMD registers
		__m256 aCoeffs = _mm256_loadu_ps(a);
		__m256 bCoeffs = _mm256_loadu_ps(b);
		__m256 cCoeffs = _mm256_loadu_ps(c);
		
		// Calculate discriminant: b² - 4ac
		// Using fused multiply-add for better precision: b*b - 4*a*c
		__m256 four = _mm256_set1_ps(4.0f);
		__m256 ac = _mm256_mul_ps(aCoeffs, cCoeffs);
		__m256 four_ac = _mm256_mul_ps(four, ac);
		__m256 b_squared = _mm256_mul_ps(bCoeffs, bCoeffs);
		__m256 discriminant = _mm256_sub_ps(b_squared, four_ac);
		
		// Create mask for discriminant >= 0 (real roots)
		__m256 zero = _mm256_setzero_ps();
		__m256 mask = _mm256_cmp_ps(discriminant, zero, _CMP_GE_OQ);
		
		// Calculate sqrt(discriminant) where discriminant >= 0
		__m256 sqrt_discriminant = _mm256_sqrt_ps(discriminant);
		
		// Calculate -b
		__m256 neg_b = _mm256_sub_ps(zero, bCoeffs);
		
		// Calculate numerator: -b - sqrt(discriminant)
		__m256 numerator = _mm256_sub_ps(neg_b, sqrt_discriminant);
		
		// Calculate denominator: 2*a
		__m256 two = _mm256_set1_ps(2.0f);
		__m256 denominator = _mm256_mul_ps(two, aCoeffs);
		
		// Calculate result: (-b - sqrt(discriminant)) / (2*a)
		__m256 solution = _mm256_div_ps(numerator, denominator);
		
		// Set NaN for complex roots (discriminant < 0)
		__m256 nan = _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN());
		__m256 final_result = _mm256_blendv_ps(nan, solution, mask);
		
		// Store result
		result.v = final_result;
	};
	
	// Run the SIMD implementation once to get the results
	simd_func();

	// Print SIMD results
	std::cout << "SIMD solutions (smaller root):" << std::endl;
	for (int lane = 0; lane < 8; ++lane) {
		std::cout << "Equation " << (lane+1) << ": ";
		if (std::isnan(result.a[lane])) {
			std::cout << "Complex roots" << std::endl;
		} else {
			std::cout << result.a[lane] << std::endl;
		}
	}
	std::cout << std::endl;

	// Benchmark comparison
	benchmark_comparison(
		"Quadratic Equation Solver",
		scalar_func,
		simd_func,
		1000000  // 1 million iterations
	);

	// Free allocated memory
	free(a);
	free(b);
	free(c);

	return 0;
}