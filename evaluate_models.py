import mlx.core as mx
import numpy as np
import time
import matplotlib.pyplot as plt

from models import TeacherCNN, StudentCNN, accuracy
from utils import load_mnist, measure_inference_time

def compare_models_on_examples(teacher, student, X_test, y_test, num_examples=5):
    # 隨機選擇一些測試樣本
    indices = np.random.choice(len(X_test), num_examples, replace=False)
    X_examples = X_test[indices]
    y_examples = y_test[indices]
    
    # 獲取模型預測
    teacher_logits = teacher(X_examples)
    student_logits = student(X_examples)
    
    teacher_probs = mx.softmax(teacher_logits, axis=1)
    student_probs = mx.softmax(student_logits, axis=1)
    
    teacher_preds = mx.argmax(teacher_probs, axis=1)
    student_preds = mx.argmax(student_probs, axis=1)
    
    # 繪製結果
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 3*num_examples))
    
    for i in range(num_examples):
        # 繪製圖像 - Access the image properly for channels-last format
        image = X_examples[i, :, :, 0]  # Extract the image without the channel dimension
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title(f"True Label: {y_examples[i]}")
        axes[i, 0].axis('off')
        
        # 繪製教師模型預測概率
        axes[i, 1].bar(range(10), teacher_probs[i])
        axes[i, 1].set_title(f"Teacher Prediction: {teacher_preds[i]}")
        axes[i, 1].set_xticks(range(10))
        
        # 繪製學生模型預測概率
        axes[i, 2].bar(range(10), student_probs[i])
        axes[i, 2].set_title(f"Student Prediction: {student_preds[i]}")
        axes[i, 2].set_xticks(range(10))
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    plt.show()

def main():
    # 加載數據
    X_train, y_train, X_test, y_test = load_mnist()
    
    # 加載模型
    teacher = TeacherCNN()
    student = StudentCNN()
    
    try:
        teacher.parameters = mx.load("models/teacher_model.npz")
        student.parameters = mx.load("models/student_model.npz")
    except:
        print("錯誤: 模型文件未找到。請先運行 train_teacher.py 和 distill_student.py")
        return
    
    # 評估模型
    teacher_acc = accuracy(teacher, X_test, y_test).item()
    student_acc = accuracy(student, X_test, y_test).item()
    
    print("=== 模型評估 ===")
    print(f"教師模型準確率: {teacher_acc:.4f}")
    print(f"學生模型準確率: {student_acc:.4f}")
    print(f"準確率差異: {teacher_acc - student_acc:.4f}")
    
    # 測量不同批次大小下的推理時間
    batch_sizes = [1, 4, 16, 64, 256]
    teacher_times = []
    student_times = []
    
    print("\n=== 不同批次大小的推理時間 ===")
    print("批次大小 | 教師模型 | 學生模型 | 速度提升")
    print("---------|----------|----------|--------")
    
    for batch_size in batch_sizes:
        # 測量教師模型時間
        t_start = time.time()
        for _ in range(5):  # 重複多次以獲得更可靠的結果
            teacher(X_test[:batch_size])
        t_time = (time.time() - t_start) / 5
        teacher_times.append(t_time)
        
        # 測量學生模型時間
        s_start = time.time()
        for _ in range(5):
            student(X_test[:batch_size])
        s_time = (time.time() - s_start) / 5
        student_times.append(s_time)
        
        speedup = t_time / s_time
        print(f"{batch_size:9d} | {t_time:.6f}s | {s_time:.6f}s | {speedup:.2f}x")
    
    # 繪製批次大小與推理時間的關係
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, teacher_times, 'o-', label='Teacher')
    plt.plot(batch_sizes, student_times, 'o-', label='Student')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Batch Size')
    plt.ylabel('Inference Time (s)')
    plt.title('Inference Time vs Batch Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('inference_time.png')
    plt.show()
    
    # 比較模型在特定例子上的預測
    compare_models_on_examples(teacher, student, X_test, y_test)

if __name__ == "__main__":
    main()
    