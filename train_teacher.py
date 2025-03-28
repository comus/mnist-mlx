import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
from sklearn.metrics import confusion_matrix
import seaborn as sns

from models import TeacherCNN, accuracy, count_parameters
from utils import load_mnist, cross_entropy_loss

# Set up font for Chinese characters
def setup_chinese_font():
    if platform.system() == 'Windows':
        font_path = 'C:/Windows/Fonts/msyh.ttc'  # Microsoft YaHei
    elif platform.system() == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/PingFang.ttc'  # PingFang
    else:  # Linux
        font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # WenQuanYi
    
    # Check if font exists
    try:
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        return True
    except:
        print("Warning: Chinese font not found. Chinese characters may not display correctly.")
        return False

# Call this function at the beginning of your script
setup_chinese_font()

# 確保模型保存目錄存在
if not os.path.exists("models"):
    os.makedirs("models")

# 訓練步驟
def train_step(model, X, y, optimizer):
    print(f"X shape before loss_fn: {X.shape}")  # Debug print
    loss_and_grad_fn = nn.value_and_grad(model, cross_entropy_loss)
    loss, grads = loss_and_grad_fn(model, X, y)
    optimizer.update(model, grads)
    return loss

# 可視化一些預測結果
def visualize_predictions(model, X_test, y_test, num_samples=10):
    # 隨機選擇一些樣本
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    # Convert NumPy indices to MLX indices
    mlx_indices = mx.array(indices)
    
    # Now use the MLX indices for indexing
    X_samples = X_test[mlx_indices]
    y_samples = y_test[mlx_indices]
    
    # 獲取預測結果
    logits = model(X_samples)
    preds = mx.argmax(logits, axis=1)
    preds_numpy = np.array(preds.tolist())
    
    # 繪製結果
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        # 獲取圖像並調整形狀以便於顯示
        img = np.array(X_samples[i].tolist())
        if len(img.shape) == 4:  # (1, C, H, W) 或 (1, H, W, C)
            img = img.squeeze()
        if len(img.shape) == 3:  # (C, H, W) 或 (H, W, C)
            if img.shape[0] == 1:  # 第一維是通道
                img = img.squeeze(0)
            elif img.shape[-1] == 1:  # 最後一維是通道
                img = img.squeeze(-1)
        
        # 顯示圖像
        axes[i].imshow(img, cmap='gray')
        y_item = y_samples[i].item()
        axes[i].set_title(f"Pred: {preds_numpy[i]}, True: {y_item}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/prediction_samples.png")
    return fig

# 繪製混淆矩陣
def plot_confusion_matrix(model, X_test, y_test):
    # 獲取預測結果 - process in batches to handle large test sets
    batch_size = 1000  # Use a smaller batch size to avoid memory issues
    all_preds = []
    
    for i in range(0, len(X_test), batch_size):
        end = min(i + batch_size, len(X_test))
        # Convert range to list first
        batch_indices = mx.array(list(range(i, end)))
        
        X_batch = X_test[batch_indices]
        logits = model(X_batch)
        preds = mx.argmax(logits, axis=1)
        all_preds.append(np.array(preds.tolist()))
    
    # Combine all predictions
    preds = np.concatenate(all_preds)
    y_true = np.array(y_test.tolist())
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, preds)
    
    # 繪製混淆矩陣
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig("results/confusion_matrix.png")
    return plt.gcf()

def main():
    # 確保結果目錄存在
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # 加載數據
    X_train, y_train, X_test, y_test = load_mnist()
    
    # 初始化教師模型
    print("\n=== 開始訓練教師模型 ===")
    teacher = TeacherCNN()
    optimizer = optim.Adam(learning_rate=0.001)
    
    # 訓練參數
    batch_size = 128
    num_epochs = 3  # 為了快速示範，使用較少的 epochs
    
    # 用於繪製訓練過程的數據
    train_losses = []
    val_accuracies = []
    epochs = []
    
    # 開始訓練
    train_start_time = time.time()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        # 創建批次索引
        indices = np.random.permutation(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            # Convert numpy indices to MLX indices
            batch_indices = mx.array(indices[i:i+batch_size])
            
            # Using advanced indexing with MLX arrays
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            print(f"X_batch shape: {X_batch.shape}")
            
            loss = train_step(teacher, X_batch, y_batch, optimizer)
            total_loss += loss.item()
            num_batches += 1
        
        # 計算平均損失和準確率
        avg_loss = total_loss / num_batches
        acc = accuracy(teacher, X_test, y_test).item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
        
        # 儲存數據用於繪圖
        train_losses.append(avg_loss)
        val_accuracies.append(acc)
        epochs.append(epoch+1)
    
    train_time = time.time() - train_start_time
    
    # 最終評估
    final_acc = accuracy(teacher, X_test, y_test).item()
    param_count = count_parameters(teacher)
    
    # 保存模型
    with open("models/teacher_model.pkl", "wb") as f:
        pickle.dump(teacher.parameters(), f)
    
    # 繪製訓練過程
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', label='Training Loss')
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'o-', label='Validation Accuracy', color='orange')
    plt.title('Validation Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/training_progress.png")
    
    # 可視化一些預測結果
    pred_fig = visualize_predictions(teacher, X_test, y_test)
    
    # 繪製混淆矩陣
    cm_fig = plot_confusion_matrix(teacher, X_test, y_test)
    
    # 輸出結果
    print("\n=== 教師模型訓練結果 ===")
    print(f"訓練時間: {train_time:.2f} 秒")
    print(f"最終準確率: {final_acc:.4f}")
    print(f"參數數量: {param_count:,}")
    print("模型已保存到 models/teacher_model.pkl")
    print("訓練進度圖表已保存到 results/training_progress.png")
    print("預測樣本圖表已保存到 results/prediction_samples.png")
    print("混淆矩陣已保存到 results/confusion_matrix.png")

if __name__ == "__main__":
    main()