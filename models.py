import mlx.core as mx
import mlx.nn as nn
import numpy as np

# 教師模型 - 較大的 CNN
class TeacherCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(5 * 5 * 64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def __call__(self, x):
        print(f"Input tensor shape in model: {x.shape}")  # Debug print
        
        # Check if tensor is properly shaped before transpose
        if len(x.shape) == 4:  # Should have 4 dimensions [batch, channels, height, width]
            x = mx.transpose(x, [0, 2, 3, 1])
        else:
            raise ValueError(f"Expected 4D input tensor, got shape {x.shape}")
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # Flattens all dimensions except batch
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        
        # If you need channels-first output, transpose back before returning
        # x = mx.transpose(x, [0, 3, 1, 2])
        return x

# 學生模型 - 較小的 CNN
class StudentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 3))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(13 * 13 * 16, 64)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
        
    def __call__(self, x):
        # If using Option 2 (channels-first format)
        x = mx.transpose(x, [0, 2, 3, 1])
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x

# 計算模型參數數量
def count_parameters(model):
    total = 0
    
    def process_param(param):
        nonlocal total
        if hasattr(param, 'size'):
            return param.size
        elif hasattr(param, 'shape'):
            return np.prod(param.shape)
        elif isinstance(param, dict):
            # 處理嵌套字典
            param_count = 0
            for key, value in param.items():
                param_count += process_param(value)
            return param_count
        else:
            print(f"Unknown parameter type: {type(param)}")
            return 0
    
    # 處理所有參數
    for key, param in model.parameters().items():
        total += process_param(param)
        
    return total

# 模型評估函數
def accuracy(model, X, y):
    logits = model(X)
    preds = mx.argmax(logits, axis=1)
    return mx.mean(preds == y)
