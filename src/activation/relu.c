#include <activation/relu.h>

float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}