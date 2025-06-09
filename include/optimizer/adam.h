#ifndef OPTIMIZER_ADAM_H
#define OPTIMIZER_ADAM_H

#include <stddef.h>

typedef struct {
    float* m;            // 动量估计
    float* v;            // 未中心化的方差估计
    size_t t;            // 时间步
    float* weights;      // 权重数组
    float* gradients;    // 梯度数组
    size_t weight_size;         // 权重数组的大小
    float learning_rate; // 学习率
    float decline_beta1;         // 衰减率1
    float decline_beta2;         // 衰减率2
    float epsilon;       // 为了数值稳定性
} Adam;

Adam* adam_init(
    size_t weight_size,
    float learning_rate, 
    float decline_beta1, 
    float decline_beta2, 
    float epsilon
);

void adam_reset_state(Adam* optimizer);
void adam_update(
    Adam* optimizer, 
    float* weights, 
    const float* gradients, 
    size_t weight_size
);

void adam_adjust_learning_rate(Adam* optimizer, float decay_rate);
void adam_free(Adam* optimizer);

#endif // OPTIMIZER_ADAM_H