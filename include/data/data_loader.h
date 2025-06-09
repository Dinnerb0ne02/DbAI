// include/data/data_loader.h
#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <stdio.h> // 用于FILE *
#include <stddef.h> // For size_t

#define MAX_LINE_LENGTH 1024
#define MAX_FLOATS_PER_LINE 4

typedef struct {
    float *features; // 存储特征的数组
    float *targets;  // 存储目标的数组（例如，标签）
    size_t num_samples; // 样本数量
    size_t num_features; // 每个样本的特征数量
    size_t num_targets;  // 每个样本的目标数量
} DataLoader;

// 创建数据加载器并从文件中加载数据
DataLoader* create_data_loader(const char* data_path, size_t num_samples, size_t num_features);

// 释放数据加载器分配的资源
void free_data_loader(DataLoader* loader);

#endif // DATA_LOADER_H