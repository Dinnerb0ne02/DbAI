#include <math.h>

#include <activation/sigmoid.h>

// Sigmoid 激活函数
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Sigmoid 导数函数
float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0 - s);
}