#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <network/layer.h>
#include <transformer/transformer.h>

Transformer* transformer_init(size_t num_layers, size_t d_model, size_t num_heads, size_t dff) {
    Transformer* transformer = malloc(sizeof(Transformer));
    transformer->num_layers = num_layers;
    transformer->blocks = malloc(num_layers * sizeof(TransformerBlock));
    transformer->d_model = d_model;
    transformer->num_heads = num_heads;
    transformer->dff = dff;

    for (size_t i = 0; i < num_layers; ++i) {
        // 创建自注意力层
        size_t qkv_size = d_model / num_heads * 3; // Q, K, V
        transformer->blocks[i].self_attention = create_layer(d_model, qkv_size);

        // 创建前馈网络层
        transformer->blocks[i].feed_forward = create_layer(d_model, dff);
    }

    return transformer;
}

void transformer_forward(Transformer* transformer, float* input, float* output) {
    memcpy(output, input, transformer->d_model * sizeof(float));

    for (size_t i = 0; i < transformer->num_layers; ++i) {
        // 自注意力
        float* self_attention_output = malloc(transformer->d_model * sizeof(float));
        layer_forward(transformer->blocks[i].self_attention, output, self_attention_output);

        // 残差连接和层归一化（省略了实现）
        // ...

        // 前馈网络
        layer_forward(transformer->blocks[i].feed_forward, self_attention_output, output);

        // 残差连接和层归一化（省略了实现）
        // ...

        free(self_attention_output); // 释放临时内存
    }
}

// 释放
void transformer_free(Transformer* transformer) {
    for (size_t i = 0; i < transformer->num_layers; ++i) {
        free_layer(transformer->blocks[i].self_attention);
        free_layer(transformer->blocks[i].feed_forward);
    }
    free(transformer->blocks);
    free(transformer);
}

