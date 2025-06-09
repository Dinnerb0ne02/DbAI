#include <stdlib.h>
#include <time.h>

#include <network/layer.h>

// 2024-07-31 18:13 improvement 1: 用 calloc() 替代 malloc() 初始化, 为 0
// 2024-07-31 18:13 improvement 2: 用 Xavier 初始化方法, 初始化权重

// 创建
Layer* create_layer(size_t input_size, size_t output_size) {
    Layer* layer = calloc(1, sizeof(Layer)); // 使用 calloc 初始化为 0
    if (!layer) return NULL;

    layer->input_size = input_size;
    layer->output_size = output_size;
    
    // 为权重和偏置分配内存
    layer->weights = calloc(input_size * output_size, sizeof(float));
    layer->biases = calloc(output_size, sizeof(float));
    if (!layer->weights || !layer->biases) {
        free_layer(layer);
        return NULL;
    }

    // 初始化权重 - 使用 Xavier
    float limit = sqrtf(1.0f / input_size); // Xavier 初始化
    for (size_t i = 0; i < input_size * output_size; ++i) {
        layer->weights[i] = ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
    }

    // 偏置初始化为 0（通过 calloc 实现）

    return layer;
}

// 释放层内存的函数保持不变
void free_layer(Layer* layer) {
    free(layer->weights);
    free(layer->biases);
    free(layer);
}

// 前向传播计算
void forward_pass(Layer* layer, const float* inputs, float* outputs) {
    // 2024-07-31 18:17 待提升: 可以使用BLAS 库, 调用 cblas_sgemm 加速矩阵乘法
    // 2024-07-31 18:17 但是我不想依赖任何外部库

    // 简单的两层矩阵乘法实现，实际中可以使用更高效的库函数
    for (size_t i = 0; i < layer->output_size; ++i) {
        for (size_t j = 0; j < layer->input_size; ++j) {
            outputs[i] += inputs[j] * layer->weights[i * layer->input_size + j];
        }
        outputs[i] += layer->biases[i];
    }
}