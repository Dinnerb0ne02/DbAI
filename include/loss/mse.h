#ifndef LOSS_MSE_H
#define LOSS_MSE_H

#include <stddef.h>

float mse(const float* predictions, const float* targets, size_t length);

#endif // LOSS_MSE_H
// include/loss/mse.h