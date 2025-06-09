#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <optimizer/adam.h>


Adam* adam_init(size_t size, float learning_rate, float beta1, float beta2, float epsilon) {
    Adam* optimizer = malloc(sizeof(Adam));
    if (!optimizer) {
        return NULL; // 检查malloc是否成功
    }

    optimizer->m = calloc(size, sizeof(float));
    if (!optimizer->m) {
        free(optimizer); // 如果calloc失败，释放之前分配的内存
        return NULL;
    }

    optimizer->v = calloc(size, sizeof(float));
    if (!optimizer->v) {
        free(optimizer->m); // 释放动量估计的内存
        free(optimizer);   // 释放优化器结构的内存
        return NULL;
    }

    optimizer->size = size; // 设置权重数组的大小
    optimizer->t = 0;
    optimizer->learning_rate = learning_rate;
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->epsilon = epsilon;

    return optimizer;
}

// 重置优化器的状态
void adam_reset_state(Adam* optimizer) {
    memset(optimizer->m, 0, optimizer->size * sizeof(float));
    memset(optimizer->v, 0, optimizer->size * sizeof(float));
    optimizer->t = 0;
}

// Adam更新公式
void adam_update(Adam* optimizer, float* weights, const float* gradients, size_t size) {
    const float beta1 = optimizer->beta1;
    const float beta2 = optimizer->beta2;
    const float epsilon = optimizer->epsilon;
    float* m = optimizer->m;
    float* v = optimizer->v;
    size_t t = optimizer->t;
    float learning_rate = optimizer->learning_rate;

    // 初始化第一个时间步
    if (t == 0) {
        for (size_t i = 0; i < size; ++i) {
            m[i] = gradients[i];
            v[i] = gradients[i] * gradients[i];
        }
    } else {
        // Adam参数更新规则
        for (size_t i = 0; i < size; ++i) {
            // 更新动量估计
            m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];
            // 更新未中心化的方差估计
            v[i] = beta2 * v[i] + (1 - beta2) * (gradients[i] * gradients[i]);

            // 计算修正后的动量和方差估计
            float bias_correction1 = 1.0f / (1.0f - powf(beta1, t));
            float bias_correction2 = 1.0f / (1.0f - powf(beta2, t));
            float m_hat = m[i] * bias_correction1;
            float v_hat = v[i] * bias_correction2;

            // 更新权重
            weights[i] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
        }
    }
    // 更新时间步
    optimizer->t = t + 1;
}

// 调整Adam优化器的学习率
void adam_adjust_learning_rate(Adam* optimizer, float decay_rate) {
    optimizer->learning_rate *= decay_rate;
}

// Adam内存释放
void adam_free(Adam* optimizer) {
    free(optimizer->m);
    free(optimizer->v);
    free(optimizer);
}