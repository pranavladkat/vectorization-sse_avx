# vectorization-sse_avx

Basic addition operation using sse and avx

compile:
- g++ -std=c++11 -mavx -msse main.cpp -o vect

output:
- sse = 0.0236244 (seconds)
- w/o sse = 0.0417704 (seconds, without vectorization)
- avx = 0.0203035 (seconds)
