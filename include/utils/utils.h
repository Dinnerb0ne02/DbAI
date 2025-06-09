#ifndef UTILS_UTILS_H
#define UTILS_UTILS_H

#include <stddef.h> // size_t
#include <loss/mse.h> //mse

#include <data/data_loader.h> // DataLoader
#include <network/network.h>


void initialize_weights(float* weights, size_t size);// 权重初始化函数

void gradient_check(
    float* predictions, 
    const float* labels, 
    float* gradients, 
    size_t num_samples, 
    size_t num_classes, 
    float (*mse_func)
        (const float*, 
        const float*, 
        size_t
    )
);
// 梯度检查函数

void shuffle_data(DataLoader* loader);

int get_batch(
    DataLoader* loader, 
    size_t batch_index, 
    float** inputs, 
    float** labels
);

void backward_pass_network(
    Network* net, 
    const float* batch_labels, 
    float learning_rate
);
// 反向传播算法, 计算梯度下降

float evaluate_network(Network* net, DataLoader* test_loader);
// 评估网络性能


#endif // UTILS_UTILS_H