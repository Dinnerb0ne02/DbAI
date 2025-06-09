#include <loss/mse.h>

float mse(const float* predictions, const float* targets, size_t length) {
    float sum = 0.0f;
    for (size_t i = 0; i < length; ++i) {
        float error = predictions[i] - targets[i];
        sum += error * error;
    }
    return sum / length;
}