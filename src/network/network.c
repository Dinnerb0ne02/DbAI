#include <stdlib.h>
#include <string.h>

#include <network/network.h>


// 创建新网络
Network* create_network(size_t num_layers) {
    if (num_layers == 0) return NULL; // 检查层数是否为0

    Network* network = malloc(sizeof(Network));
    if (network == NULL) return NULL; // 检查内存分配

    network->layers = malloc(sizeof(Layer*) * num_layers);
    if (network->layers == NULL) {
        free(network);
        return NULL; // 检查内存分配
    }

    network->num_layers = num_layers;
    return network;
}

// 释放网络资源
void free_network(Network* network) {
    if (network == NULL) return; // 检查空指针

    for (size_t i = 0; i < network->num_layers; ++i) {
        free_layer(network->layers[i]); // 释放每层资源
    }
    free(network->layers); // 释放层指针数组
    free(network); // 释放网络结构
}

// 添加层到网络
int add_layer(Network* network, size_t input_size, size_t output_size) {
    if (network == NULL || input_size == 0 || output_size == 0) return -1; // 检查无效输入

    Layer* layer = create_layer(input_size, output_size);
    if (layer == NULL) return -1; // 检查层是否创建

    network->layers[network->num_layers++] = layer; // 添加层到网络
    return 0; // 正常
}

// 网络前向传播
void forward_pass_network(Network* network, const float* inputs, float* outputs) {
    if (network == NULL || inputs == NULL || outputs == NULL) return; // 检查空指针

    size_t output_size = network->layers[0]->output_size;
    if (output_size == 0) return; // 检查输出尺寸

    float* current_output = malloc(sizeof(float) * output_size);
    if (current_output == NULL) return; // 检查内存分配

    // 第一层前向传播
    forward_pass(network->layers[0], inputs, current_output);

    //行剩余层前向传播
    for (size_t i = 1; i < network->num_layers; ++i) {
        output_size = network->layers[i]->output_size;
        float* next_output = realloc(current_output, sizeof(float) * output_size);
        if (next_output == NULL) break; // 检查内存,  重新分配
        current_output = next_output;

        forward_pass(network->layers[i], current_output, current_output);
    }

    memcpy(outputs, current_output, sizeof(float) * network->layers[network->num_layers - 1]->output_size);
    free(current_output); // 释放中间输出数组
}