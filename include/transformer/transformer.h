#ifndef TRANSFORMER_TRANSFORMER_H
#define TRANSFORMER_TRANSFORMER_H

#include <stddef.h> // size_t

#include <network/layer.h>

typedef struct {
    Layer* self_attention;
    Layer* feed_forward;
} TransformerBlock;


typedef struct {
    size_t num_layers;
    Layer** layers;
    size_t d_model; // 模型的维度
    size_t num_heads; // 注意力头的数量
    size_t dff; // 馈前网络的维度
    TransformerBlock* blocks;
} Transformer;

Transformer* transformer_init(
    size_t num_layers, 
    size_t d_model,
    size_t num_heads,
    size_t dff
);

void transformer_forward(
    Transformer* transformer, 
    const float* input,
    float* output
); // 向前传播函数

void add_layer(Transformer* transformer, Layer* layer); // 辅助函数声明

void transformer_free(Transformer* transformer);

void layer_forward(
    Layer* layer, 
    const float* input, 
    float* output
); // 辅助函数声明
#endif