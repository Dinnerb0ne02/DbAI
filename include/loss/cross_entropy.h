#ifndef LOSS_CROSS_ENTROPY_H
#define LOSS_CROSS_ENTROPY_H

#include <stddef.h>

float cross_entropy(const float* predictions, const float* targets, size_t length);

#endif // LOSS_CROSS_ENTROPY_H