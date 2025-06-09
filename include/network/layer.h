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

void layer_forward(
    Layer* self_attention, 
    const float* input, 
    float* self_attention_output
);
void backward_pass(Layer* layer, const float* input_gradients, const float* output_gradients);

void initialize_weights(float* weights, size_t size);
void initialize_biases(float* biases, size_t size);
void initialize_gradients(float* gradients, size_t size);
#endif // NETWORK_LAYER_H