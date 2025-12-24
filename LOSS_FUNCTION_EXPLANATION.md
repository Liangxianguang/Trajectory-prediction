# 损失函数说明

## 使用的损失函数：Mean Squared Error (MSE)

### 数学公式

$$\text{MSE Loss} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

其中：
- $y_i$ 是真实值（真实轨迹点）
- $\hat{y}_i$ 是预测值（模型预测的轨迹点）
- $n$ 是样本数量

### 为什么使用 MSE？

1. **有效捕捉误差**：MSE 有效地捕捉了预测轨迹和实际轨迹点之间的平方差异

2. **强调大误差**：平方操作对较大的误差施加更高的惩罚，使模型更加关注显著的偏差

3. **数值稳定性**：MSE 在优化过程中表现稳定，梯度计算简单直接

4. **适用于连续值**：轨迹预测涉及连续的坐标值，MSE 特别适合这类问题

### 实现细节

在 `train_model_enhanced.py` 中的 `MultiObjectiveLoss` 类实现了 MSE 损失：

```python
class MultiObjectiveLoss(nn.Module):
    """Mean Squared Error (MSE) 损失函数"""
    
    def forward(self, pred, target, plane_preds=None):
        """
        计算 MSE 损失
        
        Args:
            pred: (batch, output_steps, 3) 预测位置增量
            target: (batch, output_steps, 3) 真实位置增量
        
        Returns:
            mse_loss: 标量
        """
        # MSE Loss = (1/n) * Σ(pred - target)²
        mse_loss = torch.mean((pred - target) ** 2)
        return mse_loss
```

### 训练过程

在训练循环中，损失函数计算如下：

1. **前向传播**：模型预测 10 步的位置增量
   - 输入：历史轨迹 (20 步)
   - 输出：未来增量 (10 步)

2. **损失计算**：计算预测增量和真实增量之间的 MSE

3. **反向传播**：根据 MSE 损失计算梯度并更新模型参数

### 预期效果

- 模型将通过最小化 MSE 损失来学习更准确的轨迹预测
- 较大的预测误差会受到更大的惩罚，促使模型关注预测精度
- 简洁的损失函数设计便于理解和调试

### 与论文的对应关系

此实现直接对应论文中的公式 (11)：

> This loss function is particularly suited for our task as it effectively captures the average 
> squared difference between the predicted and actual trajectory points, emphasizing more 
> significant errors.

---

**模型状态**：✅ 已更新为使用纯 MSE 损失函数
