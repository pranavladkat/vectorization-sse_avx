#include <iostream>
#include <xmmintrin.h>
#include <immintrin.h>
#include <chrono>

void sse_add(float* x, float* y,float* z,size_t n){
    __m128 a,b;
    for(size_t i = 0; i < n; i+=4){
        a = _mm_load_ps(x+i);
        b = _mm_load_ps(y+i);
        b = _mm_add_ps(a,b);
        _mm_store_ps(z+i,b);
    }

    // operate on remaining data
    for(size_t i = n - n%4; i < n; ++i)
        z[i] = x[i] + y[i];
}

void avx_add(float* x, float* y,float* z,size_t n){
    __m256 a,b;
    for(size_t i = 0; i < n; i+=8){
        a = _mm256_load_ps(x+i);
        b = _mm256_load_ps(y+i);
        b = _mm256_add_ps(a,b);
        _mm256_store_ps(z+i,b);
    }

    // operate on remaining data
    for(size_t i = n - n%8; i < n; ++i)
        z[i] = x[i] + y[i];
}


int main(){

    size_t n = 12800000;
    // sse requires 16 bytes aligned memory
    float* x = (float*)_mm_malloc(n*sizeof(float),16);
    float* y = (float*)_mm_malloc(n*sizeof(float),16);
    float* z = (float*)_mm_malloc(n*sizeof(float),16);

    for(size_t i = 0; i < n; ++i){
        x[i] = i*0.45;
        y[i] = 1 + i*0.76;
        z[i] = 0.;
    }

    auto start = std::chrono::system_clock::now();
    sse_add(x,y,z,n);
    std::chrono::duration<double> time = std::chrono::system_clock::now() - start;
    std::cout << "sse = " << time.count() << std::endl;

    start = std::chrono::system_clock::now();
    for(size_t i = 0; i < n; ++i){
        z[i] = x[i] + y[i];
    }
    time = std::chrono::system_clock::now() - start;
    std::cout << "w/o sse = " << time.count() << std::endl;

    _mm_free(x);
    _mm_free(y);
    _mm_free(z);

    // sse requires 32 bytes aligned memory
    x = (float*)_mm_malloc(n*sizeof(float),32);
    y = (float*)_mm_malloc(n*sizeof(float),32);
    z = (float*)_mm_malloc(n*sizeof(float),32);

    for(size_t i = 0; i < n; ++i){
        x[i] = i*0.45;
        y[i] = 1 + i*0.76;
        z[i] = 0.;
    }

    start = std::chrono::system_clock::now();
    avx_add(x,y,z,n);
    time = std::chrono::system_clock::now() - start;
    std::cout << "avx = " << time.count() << std::endl;

    _mm_free(x);
    _mm_free(y);
    _mm_free(z);

    return 0;
}
