#ifndef _RANDOM_NUMBER_GENERATOR_
#define _RANDOM_NUMBER_GENERATOR_

#include <random>
#include <cassert>
#include <iostream>
#include "utils.h"

template <typename T>
void randn(float *mat, int num) {
    std::random_device rd;   // 随机数种子
    std::mt19937 gen(rd());  // 随机数生成器
    std::normal_distribution<T> dist(0.0, 1.0);
    // std::uniform_real_distribution<T> dist(1.0, 2.0);

    for (int i = 0; i < num; ++i) {
        mat[i] = dist(gen);
    }
}

#endif  // _RANDOM_NUMBER_GENERATOR_