#ifndef NETWORK_NETWORK_H
#define NETWORK_NETWORK_H

#include <stddef.h>

#include <network/layer.h>

typedef struct {
    Layer** layers; // 网络中的层
    size_t num_layers; // 层的数量
} Network;

// 函数原型声明
Network* create_network(size_t num_layers);
void free_network(Network* network);
int add_layer(Network* network, size_t input_size, size_t output_size);
void forward_pass_network(Network* network, const float* inputs, float* outputs);

#endif // NETWORK_NETWORK_H