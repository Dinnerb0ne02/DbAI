#ifndef OPTIMIZER_SGD_H
#define OPTIMIZER_SGD_H

#include <stddef.h>

typedef struct {
    float learning_rate;
} SGD;

void sgd_reset_state(SGD* optimizer);
void sgd_update(float* weights, float* gradients, size_t size, float learning_rate);
void sgd_adjust_learning_rate(SGD* optimizer, float decay_rate);

#endif // OPTIMIZER_SGD_H