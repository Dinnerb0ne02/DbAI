#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <utils/utils.h>
#include <data/data_loader.h>
#include <network/network.h>

#define BATCH_SIZE 32

void initialize_weights(float* weights, size_t size) {
    srand(time(NULL));
    for (size_t i = 0; i < size; ++i) {
        weights[i] = (rand() / (float)RAND_MAX) * 0.1 - 0.05; 
        // 初始化为 [-0.05, 0.05] 区间的随机数
    }
}

void gradient_check(
    float* predictions, 
    const float* labels, 
    float* gradients, 
    size_t num_samples, 
    size_t num_classes, 
    float (*mse_func)(const float*, const float*, size_t)
) {
    const float epsilon = 1e-5; // 用于数值梯度的微小变化量epsilon
    float grad_approx, diff;
    for (size_t s = 0; s < num_samples; ++s) {
        for (size_t c = 0; c < num_classes; ++c) {
            float original = predictions[s * num_classes + c];
            predictions[s * num_classes + c] = original + epsilon;
            float loss_plus_epsilon = mse_func(predictions, labels, num_samples);
            predictions[s * num_classes + c] = original - epsilon;
            float loss_minus_epsilon = mse_func(predictions, labels, num_samples);
            predictions[s * num_classes + c] = original;

            grad_approx = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon);
            diff = fabs(grad_approx - gradients[s * num_classes + c]);
            if (diff > 1e-4) {
                printf("Gradient check failed for sample %zu, class %zu. Difference: %f\n", s, c, diff);
            }
        }
    }
}

void shuffle_data(DataLoader* loader) {
    size_t n = loader->num_samples;
    for (size_t i = 0; i < n - 1; ++i) {
        size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
        float temp = loader->features[i];
        loader->features[i] = loader->features[j];
        loader->features[j] = temp;
    }
}

// 实现获取批次数据逻辑
int get_batch(
    DataLoader* loader, 
    size_t batch_index, 
    float** inputs, 
    float** labels
) {
    const size_t start = batch_index * BATCH_SIZE;
    const size_t end = start + BATCH_SIZE;
    if (end > loader->num_samples) return 0; // 检查批次是否超出数据范围

    // 动态分配批次内存
    *inputs = malloc(BATCH_SIZE * loader->num_features * sizeof(float));
    *labels = malloc(BATCH_SIZE * sizeof(float));

    if (!*inputs || !*labels) {
        // 内存分配失败，清理并返回
        free(*inputs);
        free(*labels);
        return 0;
    }

    // 复制批次数据
    for (size_t i = start, j = 0; i < end; ++i, ++j) {
        memcpy(*inputs + j * loader->num_features, loader->features + i * loader->num_features, loader->num_features * sizeof(float));
        // 假设标签是连续存储的
        (*labels)[j] = loader->targets[i];
    }
    return 1;
}

void backward_pass_network(
    Network* net, 
    const float* batch_labels, 
    float learning_rate
) {
    // 这里需要实现反向传播算法来计算每个层的梯度
    // 这通常涉及到链式法则，从输出层到输入层依次计算梯度
    // 由于这是一个复杂的过程，具体实现取决于网络结构和激活函数

    // 以下是一个简化的示例，只实现了输出层的梯度计算
    for (int i = net->num_layers - 1; i >= 0; --i) {
        // 假设我们有一个函数来计算当前层的梯度
        // 这需要根据损失函数的导数和前一层的输出来实现
        // float* gradients = calculate_gradients_for_layer(i, batch_labels, ...);
        // 然后将计算得到的梯度复制到网络层的 gradients 数组中
        // memcpy(net->layers[i]->gradients, gradients, sizeof(float) * ...);
    }
}

// 评估网络性能的函数
float evaluate_network(Network* net, DataLoader* test_loader) {
    size_t num_correct = 0;
    float* outputs = malloc(test_loader->num_samples * net->layers[net->num_layers - 1]->output_size * sizeof(float));

    for (size_t i = 0; i < test_loader->num_samples; ++i) {
        // 前向传播获取预测结果
        forward_pass_network(net, test_loader->features + i * test_loader->num_features, outputs + i * net->layers[net->num_layers - 1]->output_size);

        // 找到预测结果中概率最高的类别
        int max_index = 0;
        for (size_t j = 1; j < net->layers[net->num_layers - 1]->output_size; ++j) {
            if (outputs[i * net->layers[net->num_layers - 1]->output_size + j] > outputs[i * net->layers[net->num_layers - 1]->output_size + max_index]) {
                max_index = (int)j;
            }
        }

        // 检查预测类别是否与真实标签匹配
        if (max_index == (int)test_loader->targets[i]) {
            num_correct++;
        }
    }

    float accuracy = (float)num_correct / test_loader->num_samples;
    free(outputs);
    return accuracy;
}