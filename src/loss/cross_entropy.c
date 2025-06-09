#include <math.h>
#include <assert.h>

#include <loss/cross_entropy.h>

/**
 * 交叉熵损失函数
 *
 * 衡量了预测概率分布与真实目标分布之间的差异
 * 该函数接受预测概率和实际目标标签作为输入, 计算出平均交叉熵损失
 * 
 * @brief 计算交叉熵损失函数 
 * @param predictions 模型输出的预测概率, 长度为length
 * @param targets     实际目标标签, 采用one-hot编码形式, 长度为length
 * @param length      样本数量
 * @return            平均交叉熵损失
 */

float cross_entropy(const float* predictions, const float* targets, size_t length) {
    // 检查输入参数的有效性
    assert(predictions != NULL && targets != NULL);
    assert(length > 0);

    float sum = 0.0f;
    for (size_t i = 0; i < length; ++i) {
        // 遍历每个样本,根据one-hot编码的目标标签计算对应的交叉熵损失
        
        // 如果目标标签为1,则损失为-log(预测概率)
        // 如果目标标签为0,则损失为-log(1-预测概率）
        int target_class = (int)targets[i];
        assert(target_class == 0 || target_class == 1);
        if (target_class == 1) {
            sum += -logf(predictions[i]);
        } else {
            sum += -logf(1.0f - predictions[i]);
        }
    }

    // 返回平均交叉熵损失
    return sum / length;
}