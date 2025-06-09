#ifndef NETWORK_LAYER_H
#define NETWORK_LAYER_H

typedef struct {
    float* weights;      // 权重数组
    float* biases;       // 偏置项
    float* gradients;
    size_t input_size;   // 输入尺寸
    size_t output_size;  // 输出尺寸
} Layer;

// 函数原型声明
Layer* create_layer(size_t input_size, size_t output_size);
void free_layer(Layer* layer);
void forward_pass(Layer* layer, const float* inputs, float* outputs);

#endif // NETWORK_LAYER_H