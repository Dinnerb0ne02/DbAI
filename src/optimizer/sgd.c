
#include <optimizer/sgd.h>

// SGD优化器更新权重
void sgd_update(
    float* weights, 
    float* gradients, 
    size_t size, 
    float learning_rate
) {
    for (size_t i = 0; i < size; ++i) {
        weights[i] -= learning_rate * gradients[i];
    }
}

// 对于SGD，如果没有额外状态，这个函数可以留空
void sgd_reset_state(SGD* optimizer) {}
