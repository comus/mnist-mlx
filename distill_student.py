import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from models import TeacherCNN, StudentCNN, accuracy, count_parameters
from utils import load_mnist, distillation_loss, measure_inference_time, plot_comparison

# 蒸餾訓練步驟
def distill_step(student, teacher, X, y, optimizer, temp=3.0, alpha=0.5):
    def loss_fn(model):
        student_logits = model(X)
        teacher_logits = mx.stop_gradient(teacher(X))
        return distillation_loss(student_logits, teacher_logits, y, temp=temp, alpha=alpha)
    
    loss_and_grad_fn = nn.value_and_grad(student, loss_fn)
    loss, grads = loss_and_grad_fn(student)
    optimizer.update(student, grads)
    return loss

# 添加新的視覺化函數
def visualize_student_predictions(student, teacher, X_test, y_test, num_samples=10):
    """比較學生模型和教師模型的預測結果"""
    # 隨機選擇一些樣本
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    mlx_indices = mx.array(indices)
    
    X_samples = X_test[mlx_indices]
    y_samples = y_test[mlx_indices]
    
    # 獲取兩個模型的預測結果
    student_logits = student(X_samples)
    teacher_logits = teacher(X_samples)
    
    student_preds = mx.argmax(student_logits, axis=1)
    teacher_preds = mx.argmax(teacher_logits, axis=1)
    
    student_probs = mx.softmax(student_logits, axis=1)
    teacher_probs = mx.softmax(teacher_logits, axis=1)
    
    # 轉換為NumPy數組以便繪圖
    student_preds_np = np.array(student_preds.tolist())
    teacher_preds_np = np.array(teacher_preds.tolist())
    student_probs_np = np.array(student_probs.tolist())
    teacher_probs_np = np.array(teacher_probs.tolist())
    
    # 繪製結果
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 3))
    
    for i in range(num_samples):
        # 顯示圖像
        img = np.array(X_samples[i].tolist())
        # 處理圖像形狀以便顯示
        if len(img.shape) == 4:
            img = img.squeeze()
        if len(img.shape) == 3:
            if img.shape[0] == 1:
                img = img.squeeze(0)
            elif img.shape[-1] == 1:
                img = img.squeeze(-1)
        
        axes[i, 0].imshow(img, cmap='gray')
        y_true = y_samples[i].item()
        axes[i, 0].set_title(f"True: {y_true}")
        axes[i, 0].axis('off')
        
        # 顯示教師模型的預測概率
        axes[i, 1].bar(range(10), teacher_probs_np[i])
        axes[i, 1].set_title(f"Teacher: {teacher_preds_np[i]}")
        axes[i, 1].set_xticks(range(10))
        
        # 顯示學生模型的預測概率
        axes[i, 2].bar(range(10), student_probs_np[i])
        axes[i, 2].set_title(f"Student: {student_preds_np[i]}")
        axes[i, 2].set_xticks(range(10))
    
    plt.tight_layout()
    plt.savefig("results/student_teacher_comparison.png")
    return fig

def plot_learning_curves(losses, accuracies, teacher_acc):
    """繪製學習曲線"""
    epochs = range(1, len(losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, 'o-', label='Distillation Loss')
    plt.title('Distillation Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, 'o-', label='Student Accuracy', color='orange')
    plt.axhline(y=teacher_acc, color='blue', linestyle='--', label=f'Teacher Accuracy: {teacher_acc:.4f}')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/student_learning_curves.png")
    return plt.gcf()

def plot_confusion_matrix(model, X_test, y_test, title, filename):
    """繪製混淆矩陣"""
    batch_size = 1000
    all_preds = []
    
    for i in range(0, len(X_test), batch_size):
        end = min(i + batch_size, len(X_test))
        batch_indices = mx.array(list(range(i, end)))
        
        X_batch = X_test[batch_indices]
        logits = model(X_batch)
        preds = mx.argmax(logits, axis=1)
        all_preds.append(np.array(preds.tolist()))
    
    preds = np.concatenate(all_preds)
    y_true = np.array(y_test.tolist())
    
    cm = confusion_matrix(y_true, preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.savefig(f"results/{filename}")
    return plt.gcf()

def compare_hard_cases(student, teacher, X_test, y_test):
    """比較兩個模型在困難樣本上的表現差異"""
    # 獲取所有預測結果
    teacher_logits = teacher(X_test)
    student_logits = student(X_test)
    
    teacher_preds = mx.argmax(teacher_logits, axis=1)
    student_preds = mx.argmax(student_logits, axis=1)
    
    # 轉換為NumPy
    teacher_preds_np = np.array(teacher_preds.tolist())
    student_preds_np = np.array(student_preds.tolist())
    y_true_np = np.array(y_test.tolist())
    
    # 找出教師正確但學生錯誤的案例
    hard_cases = np.where((teacher_preds_np == y_true_np) & (student_preds_np != y_true_np))[0]
    
    if len(hard_cases) == 0:
        print("沒有找到教師正確但學生錯誤的案例!")
        return None
    
    # 選擇最多10個案例
    num_cases = min(10, len(hard_cases))
    selected_indices = hard_cases[:num_cases]
    mlx_indices = mx.array(selected_indices)
    
    X_samples = X_test[mlx_indices]
    y_samples = y_test[mlx_indices]
    
    # 獲取預測
    s_logits = student(X_samples)
    t_logits = teacher(X_samples)
    s_preds = mx.argmax(s_logits, axis=1)
    t_preds = mx.argmax(t_logits, axis=1)
    
    # 轉換為NumPy
    s_preds_np = np.array(s_preds.tolist())
    t_preds_np = np.array(t_preds.tolist())
    
    # 繪製結果
    fig, axes = plt.subplots(num_cases, 1, figsize=(8, 3 * num_cases))
    if num_cases == 1:
        axes = [axes]
        
    for i in range(num_cases):
        img = np.array(X_samples[i].tolist())
        if len(img.shape) == 4:
            img = img.squeeze()
        if len(img.shape) == 3:
            if img.shape[0] == 1:
                img = img.squeeze(0)
            elif img.shape[-1] == 1:
                img = img.squeeze(-1)
        
        axes[i].imshow(img, cmap='gray')
        y_true = y_samples[i].item()
        axes[i].set_title(f"True: {y_true}, Teacher: {t_preds_np[i]}, Student: {s_preds_np[i]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/hard_cases.png")
    return fig

def main():
    # 確保教師模型已訓練
    if not os.path.exists("models/teacher_model.pkl"):
        print("錯誤: 教師模型不存在。請先運行 train_teacher.py")
        return
    
    # 確保結果目錄存在
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # 加載數據
    X_train, y_train, X_test, y_test = load_mnist()
    
    # 加載教師模型
    teacher = TeacherCNN()
    with open("models/teacher_model.pkl", "rb") as f:
        params = pickle.load(f)
        teacher.update(params)
    
    teacher_acc = accuracy(teacher, X_test, y_test).item()
    print(f"加載教師模型，準確率: {teacher_acc:.4f}")
    
    # 初始化學生模型
    print("\n=== 開始進行知識蒸餾 ===")
    student = StudentCNN()
    optimizer = optim.Adam(learning_rate=0.001)
    
    # 蒸餾參數
    batch_size = 128
    num_epochs = 5  # 學生模型可能需要更多 epochs
    temp = 3.0      # 溫度參數
    alpha = 0.5     # 軟標籤和硬標籤的權重比例
    
    # 記錄訓練指標
    train_losses = []
    student_accuracies = []
    
    # 開始蒸餾訓練
    distill_start_time = time.time()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        # 創建批次索引
        indices = np.random.permutation(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_indices = mx.array(indices[i:i+batch_size])
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            loss = distill_step(student, teacher, X_batch, y_batch, optimizer, temp=temp, alpha=alpha)
            total_loss += loss.item()
            num_batches += 1
        
        # 計算平均損失和準確率
        avg_loss = total_loss / num_batches
        student_acc = accuracy(student, X_test, y_test).item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Student Accuracy: {student_acc:.4f}, Teacher Accuracy: {teacher_acc:.4f}")
        
        # 記錄訓練指標
        train_losses.append(avg_loss)
        student_accuracies.append(student_acc)
    
    distill_time = time.time() - distill_start_time
    
    # 最終評估
    student_acc = accuracy(student, X_test, y_test).item()
    
    # 保存學生模型
    with open("models/student_model.pkl", "wb") as f:
        pickle.dump(student.parameters(), f)
    
    # 輸出結果
    print("\n=== 知識蒸餾結果 ===")
    print(f"蒸餾時間: {distill_time:.2f} 秒")
    print(f"學生模型準確率: {student_acc:.4f}")
    print(f"教師模型準確率: {teacher_acc:.4f}")
    print("學生模型已保存到 models/student_model.pkl")
    
    # 比較模型大小
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    print("\n=== 模型對比 ===")
    print(f"教師模型參數數量: {teacher_params:,}")
    print(f"學生模型參數數量: {student_params:,}")
    print(f"縮小比例: {teacher_params / student_params:.2f}x")
    
    # 比較推理速度
    print("\n=== 推理速度對比 ===")
    teacher_infer_time = measure_inference_time(teacher, X_test, y_test)
    student_infer_time = measure_inference_time(student, X_test, y_test)
    
    print(f"教師模型推理速度: {teacher_infer_time:.4f} 秒")
    print(f"學生模型推理速度: {student_infer_time:.4f} 秒")
    print(f"速度提升: {teacher_infer_time / student_infer_time:.2f}x")
    
    # 可視化比較結果
    plot_comparison(teacher_acc, student_acc, teacher_params, student_params, 
                    teacher_infer_time, student_infer_time)

    # === 新增的視覺化 ===
    # 1. 學習曲線
    print("正在生成學習曲線圖...")
    learning_curves = plot_learning_curves(train_losses, student_accuracies, teacher_acc)
    
    # 2. 學生與教師的預測比較
    print("正在生成學生與教師的預測比較圖...")
    pred_comparison = visualize_student_predictions(student, teacher, X_test, y_test)
    
    # 3. 學生模型的混淆矩陣
    print("正在生成學生模型的混淆矩陣...")
    student_cm = plot_confusion_matrix(student, X_test, y_test, "Student Model Confusion Matrix", "student_confusion_matrix.png")
    
    # 4. 困難案例分析
    print("正在分析困難案例...")
    hard_cases_fig = compare_hard_cases(student, teacher, X_test, y_test)
    
    print("\n=== 視覺化結果 ===")
    print("模型比較圖表已保存到 results/distillation_results.png")
    print("學習曲線已保存到 results/student_learning_curves.png")
    print("學生與教師的預測比較已保存到 results/student_teacher_comparison.png")
    print("學生模型的混淆矩陣已保存到 results/student_confusion_matrix.png")
    if hard_cases_fig:
        print("困難案例分析已保存到 results/hard_cases.png")

if __name__ == "__main__":
    main()
