#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <data/data_loader.h>

/**
 * 创建数据加载器，从指定的文件路径加载训练和测试数据
 * @param train_data_path 训练数据文件的路径
 * @param test_data_path 测试数据文件的路径
 * @return 成功时返回 DataLoader*, 失败时返回 NULL
 */


// 辅助函数，解析单行数据并填充浮点数数组
static int parse_line(const char* line, float* values, size_t max_values) {
    int count = 0;
    char* token = strtok((char*)line, " \n"); // 将字符串分割并获取第一个token
    while (token != NULL && count < max_values) {
        values[count++] = atof(token); // 将字符串转换为浮点数
        token = strtok(NULL, " \n"); // 获取下一个token
    }
    return count; // 返回实际读取浮点数的数量
}

// 创建数据加载器
DataLoader* create_data_loader(const char* data_path, size_t num_samples, size_t num_features) {
    FILE *file = fopen(data_path, "r");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }

    // 检查是否有足够的样本和特征数量
    if (num_samples == 0 || num_features == 0) {
        fprintf(stderr, "Invalid number of samples or features\n");
        fclose(file);
        return NULL;
    }

    DataLoader* loader = malloc(sizeof(DataLoader));
    if (!loader) {
        perror("Memory allocation for DataLoader failed");
        fclose(file);
        return NULL;
    }

    // 为特征分配内存
    loader->features = malloc(num_samples * num_features * sizeof(float));
    if (!loader->features) {
        perror("Memory allocation for features failed");
        free(loader);
        fclose(file);
        return NULL;
    }

    loader->num_samples = num_samples;
    loader->num_features = num_features;

    char line[MAX_LINE_LENGTH];
    float values[MAX_FLOATS_PER_LINE];
    size_t features_read = 0;

    // 读取文件并且填充数据
    for (size_t i = 0; i < num_samples; ++i) {
        if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
            fprintf(stderr, "Error reading line %zu from file\n", i + 1);
            free(loader->features);
            free(loader);
            fclose(file);
            return NULL;
        }

        size_t count = parse_line(line, values, MAX_FLOATS_PER_LINE);
        if (count > num_features) {
            fprintf(stderr, "Too many values on line %zu (max %zu)\n", i + 1, num_features);
            free(loader->features);
            free(loader);
            fclose(file);
            return NULL;
        }

        // 将解析的值传递到features中
        for (size_t j = 0; j < count; ++j) {
            loader->features[features_read + j] = values[j];
        }
        features_read += count;
    }

    fclose(file);
    return loader;
}

// 释放文件加载器的内存
void free_data_loader(DataLoader* loader) {
    if (loader) {
        free(loader->features);
        free(loader);
    }
}