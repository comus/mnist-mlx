import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import os

# 數據加載函數
def load_mnist():
    print("開始加載 MNIST 數據集...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X / 255.0  # 歸一化
    
    # Reshape directly to channels-first format [N, C, H, W]
    X = X.reshape(-1, 1, 28, 28)
    y = y.astype(np.int32)
    
    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 轉換為 MLX 數組
    X_train = mx.array(X_train)
    y_train = mx.array(y_train)
    X_test = mx.array(X_test)
    y_test = mx.array(y_test)
    
    print(f"數據加載完成! 訓練集: {X_train.shape}, 測試集: {X_test.shape}")
    return X_train, y_train, X_test, y_test

# 基本損失函數
def cross_entropy_loss(model, X, y):
    logits = model(X)
    # Make sure this returns a scalar (mean or sum of losses)
    return mx.mean(nn.losses.cross_entropy(logits, y))

# 知識蒸餾損失函數
def distillation_loss(student_logits, teacher_logits, targets, temp=3.0, alpha=0.5):
    # 軟目標損失 (蒸餾損失)
    soft_targets = nn.softmax(teacher_logits / temp, axis=1)
    soft_prob = nn.softmax(student_logits / temp, axis=1)
    soft_targets_loss = mx.mean(-mx.sum(soft_targets * mx.log(soft_prob + 1e-8), axis=1)) * (temp * temp)
    
    # 硬目標損失 (交叉熵)
    hard_loss = mx.mean(nn.losses.cross_entropy(student_logits, targets))
    
    # 總損失 = 軟目標損失 * 權重 + 硬目標損失 * (1-權重)
    return alpha * soft_targets_loss + (1.0 - alpha) * hard_loss

# 結果可視化
def plot_comparison(teacher_acc, student_acc, teacher_params, student_params, teacher_time, student_time):
    """結果可視化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 準確率對比
    models = ['Teacher', 'Student']
    accuracies = [teacher_acc, student_acc]
    ax1.bar(models, accuracies, color=['blue', 'orange'])
    ax1.set_ylim([0, 1])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    
    # 在條形上添加準確率值
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    # 參數數量和時間對比
    ax2.bar(models, [teacher_time, student_time], color=['blue', 'orange'], alpha=0.7, label='Inference Time (s)')
    
    # 添加第二個 y 軸
    ax3 = ax2.twinx()
    ax3.bar(models, [teacher_params/1e6, student_params/1e6], color=['blue', 'orange'], alpha=0.3, label='Parameters (M)')
    
    ax2.set_title('Model Efficiency')
    ax2.set_ylabel('Inference Time (s)')
    ax3.set_ylabel('Parameters (Millions)')
    
    # 在條形上添加值
    for i, v in enumerate([teacher_time, student_time]):
        ax2.text(i, v + 0.001, f'{v:.4f}s', ha='center')
    
    for i, v in enumerate([teacher_params/1e6, student_params/1e6]):
        ax3.text(i, v + 0.1, f'{v:.2f}M', ha='center')
    
    plt.tight_layout()
    
    # 移除 plt.show() 調用，它會清空圖表
    plt.savefig("results/distillation_results.png")
    
    # 返回圖表對象，以便外部代碼可以進一步操作它
    return fig

# 測量推理時間
def measure_inference_time(model, X_test, y_test, repeats=10):
    start_time = time.time()
    for _ in range(repeats):
        model(X_test[:100])  # 使用較小的批次以加快測試
    return (time.time() - start_time) / repeats
