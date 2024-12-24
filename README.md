# CIFAR-10 Image Classification 🖼️🤖


This project provides a PyTorch implementation of a custom CNN architecture for CIFAR-10 image classification, designed with specific architectural constraints.

## 🎯 Model Requirements
- Has the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead)
- Total RF must be more than 44
- One of the layers must use Depthwise Separable Convolution
- One of the layers must use Dilated Convolution
- Use GAP (compulsory):- add FC after GAP to target #of classes (optional)
- use augmentation library and apply:
    - horizontal flip
    - shiftScaleRotate
    - coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
    - achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.


## 📂Project Structure 

```bash
project_root/
├── src/
│ └── model.py       # Model architecture
│ └── dataset.py     # Dataset and data loading utilities
│ └── utils.py       # Model Summary, Training & testing functions
│ └── config.py      # Configuration parameters
│ ├── train.py       # Main training script
├── requirements.txt # Project dependencies
└── README.md        # Project documentation
```

## 🛠Requirements

- Python 3.7+ 🐍
- PyTorch 1.7+ ⚡
- Albumentations 📸
- torchsummary 📊
- numpy 🔢


## 🚀Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🏗️Model Details

### Architecture
```bash
CIFAR10Net(
  (c1): Sequential(
    (0): Conv2d(3, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(24, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
  )
  (c2): Sequential(
    (0): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
    (1): Conv2d(48, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (2): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
    (4): Conv2d(48, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (5): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
  )
  (c3): Sequential(
    (0): Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
  )
  (c4): Sequential(
    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
  )
  (gap): AdaptiveAvgPool2d(output_size=1)
  (fc): Linear(in_features=64, out_features=64, bias=False)
  (dropout): Dropout(p=0.05, inplace=False)
  (fc2): Linear(in_features=64, out_features=10, bias=False)
)
```

### Parameters
```bash
==================================================
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 24, 32, 32]             648
       BatchNorm2d-2           [-1, 24, 32, 32]              48
              ReLU-3           [-1, 24, 32, 32]               0
            Conv2d-4           [-1, 48, 16, 16]          10,368
       BatchNorm2d-5           [-1, 48, 16, 16]              96
              ReLU-6           [-1, 48, 16, 16]               0
            Conv2d-7           [-1, 48, 16, 16]             864
            Conv2d-8           [-1, 48, 16, 16]           2,304
       BatchNorm2d-9           [-1, 48, 16, 16]              96
             ReLU-10           [-1, 48, 16, 16]               0
           Conv2d-11             [-1, 48, 8, 8]          20,736
      BatchNorm2d-12             [-1, 48, 8, 8]              96
             ReLU-13             [-1, 48, 8, 8]               0
           Conv2d-14             [-1, 64, 8, 8]          27,648
      BatchNorm2d-15             [-1, 64, 8, 8]             128
             ReLU-16             [-1, 64, 8, 8]               0
           Conv2d-17             [-1, 64, 4, 4]          36,864
      BatchNorm2d-18             [-1, 64, 4, 4]             128
             ReLU-19             [-1, 64, 4, 4]               0
           Conv2d-20             [-1, 64, 4, 4]          36,864
      BatchNorm2d-21             [-1, 64, 4, 4]             128
             ReLU-22             [-1, 64, 4, 4]               0
           Conv2d-23             [-1, 64, 2, 2]          36,864
      BatchNorm2d-24             [-1, 64, 2, 2]             128
             ReLU-25             [-1, 64, 2, 2]               0
AdaptiveAvgPool2d-26             [-1, 64, 1, 1]               0
           Linear-27                   [-1, 64]           4,096
          Dropout-28                   [-1, 64]               0
           Linear-29                   [-1, 10]             640
================================================================
Total params: 178,744
Trainable params: 178,744
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.44
Params size (MB): 0.68
Estimated Total Size (MB): 2.13
----------------------------------------------------------------
==================================================

Total Parameters: 178,744
Trainable Parameters: 178,744
==================================================
```


### 🏋️‍♂️ Training

```bash
python train.py
```

## 🧑‍🔬 Data Augmentation
The following augmentations are applied using the Albumentations library:

- 🤸‍♂️ Horizontal Flip (p=0.5)
- 🔄 ShiftScaleRotate
- 🧩 CoarseDropout with specific parameters
- 🌈 Normalization


## ⚙️ Model Configuration

You can modify key hyperparameters in the config.py file:

- 🧑‍🏫 Batch size
- ⚡ Learning rate
- 🔁 Number of epochs
- 🚀 Momentum
- 🏋️‍♀️ Weight decay
- 💧 Dropout value

## 📜 Training Logs

<details>
  <summary>Click to expand training logs</summary>
  
```bash
2024-12-24 03:16:21,846 - INFO - CIFAR-10 dataset status: Found
2024-12-24 03:16:23,406 - INFO - Dataloaders initialized with batch size: 32
2024-12-24 03:16:23,407 - INFO - Training config - LR: 0.015, Momentum: 0.9, Weight Decay: 1e-05
2024-12-24 03:16:23,407 - INFO - Starting training for 150 epochs
2024-12-24 03:16:23,407 - INFO - 
Epoch 1/150
2024-12-24 03:16:54,343 - INFO - Training Accuracy: 38.17%
2024-12-24 03:16:56,269 - INFO - Test Accuracy: 51.32%
2024-12-24 03:16:56,277 - INFO - New Best Test Accuracy: 51.32%
2024-12-24 03:16:56,277 - INFO - 
Epoch 2/150
2024-12-24 03:17:26,670 - INFO - Training Accuracy: 50.72%
2024-12-24 03:17:28,690 - INFO - Test Accuracy: 59.77%
2024-12-24 03:17:28,697 - INFO - New Best Test Accuracy: 59.77%
2024-12-24 03:17:28,697 - INFO - 
Epoch 3/150
2024-12-24 03:17:54,304 - INFO - Training Accuracy: 56.73%
2024-12-24 03:17:56,371 - INFO - Test Accuracy: 66.64%
2024-12-24 03:17:56,378 - INFO - New Best Test Accuracy: 66.64%
2024-12-24 03:17:56,378 - INFO - 
Epoch 4/150
2024-12-24 03:18:26,083 - INFO - Training Accuracy: 61.15%
2024-12-24 03:18:28,188 - INFO - Test Accuracy: 68.95%
2024-12-24 03:18:28,195 - INFO - New Best Test Accuracy: 68.95%
2024-12-24 03:18:28,195 - INFO - 
Epoch 5/150
2024-12-24 03:18:57,975 - INFO - Training Accuracy: 63.95%
2024-12-24 03:19:00,070 - INFO - Test Accuracy: 72.94%
2024-12-24 03:19:00,077 - INFO - New Best Test Accuracy: 72.94%
2024-12-24 03:19:00,077 - INFO - 
Epoch 6/150
2024-12-24 03:19:30,605 - INFO - Training Accuracy: 65.98%
2024-12-24 03:19:32,723 - INFO - Test Accuracy: 74.02%
2024-12-24 03:19:32,730 - INFO - New Best Test Accuracy: 74.02%
2024-12-24 03:19:32,730 - INFO - 
Epoch 7/150
2024-12-24 03:20:02,376 - INFO - Training Accuracy: 67.60%
2024-12-24 03:20:04,589 - INFO - Test Accuracy: 74.58%
2024-12-24 03:20:04,597 - INFO - New Best Test Accuracy: 74.58%
2024-12-24 03:20:04,597 - INFO - 
Epoch 8/150
2024-12-24 03:20:34,534 - INFO - Training Accuracy: 68.75%
2024-12-24 03:20:36,617 - INFO - Test Accuracy: 76.37%
2024-12-24 03:20:36,624 - INFO - New Best Test Accuracy: 76.37%
2024-12-24 03:20:36,624 - INFO - 
Epoch 9/150
2024-12-24 03:21:04,342 - INFO - Training Accuracy: 70.24%
2024-12-24 03:21:06,272 - INFO - Test Accuracy: 75.21%
2024-12-24 03:21:06,272 - INFO - 
Epoch 10/150
2024-12-24 03:21:36,877 - INFO - Training Accuracy: 70.82%
2024-12-24 03:21:38,869 - INFO - Test Accuracy: 77.47%
2024-12-24 03:21:38,876 - INFO - New Best Test Accuracy: 77.47%
2024-12-24 03:21:38,876 - INFO - 
Epoch 11/150
2024-12-24 03:22:11,208 - INFO - Training Accuracy: 71.51%
2024-12-24 03:22:13,127 - INFO - Test Accuracy: 77.87%
2024-12-24 03:22:13,134 - INFO - New Best Test Accuracy: 77.87%
2024-12-24 03:22:13,134 - INFO - 
Epoch 12/150
2024-12-24 03:22:42,926 - INFO - Training Accuracy: 72.67%
2024-12-24 03:22:44,949 - INFO - Test Accuracy: 77.39%
2024-12-24 03:22:44,949 - INFO - 
Epoch 13/150
2024-12-24 03:23:11,637 - INFO - Training Accuracy: 73.15%
2024-12-24 03:23:13,579 - INFO - Test Accuracy: 78.42%
2024-12-24 03:23:13,586 - INFO - New Best Test Accuracy: 78.42%
2024-12-24 03:23:13,586 - INFO - 
Epoch 14/150
2024-12-24 03:23:41,317 - INFO - Training Accuracy: 73.59%
2024-12-24 03:23:43,426 - INFO - Test Accuracy: 78.50%
2024-12-24 03:23:43,434 - INFO - New Best Test Accuracy: 78.50%
2024-12-24 03:23:43,434 - INFO - 
Epoch 15/150
2024-12-24 03:24:13,668 - INFO - Training Accuracy: 73.98%
2024-12-24 03:24:15,753 - INFO - Test Accuracy: 79.10%
2024-12-24 03:24:15,760 - INFO - New Best Test Accuracy: 79.10%
2024-12-24 03:24:15,760 - INFO - 
Epoch 16/150
2024-12-24 03:24:43,213 - INFO - Training Accuracy: 74.67%
2024-12-24 03:24:45,198 - INFO - Test Accuracy: 79.61%
2024-12-24 03:24:45,205 - INFO - New Best Test Accuracy: 79.61%
2024-12-24 03:24:45,205 - INFO - 
Epoch 17/150
2024-12-24 03:25:14,693 - INFO - Training Accuracy: 74.76%
2024-12-24 03:25:16,728 - INFO - Test Accuracy: 80.17%
2024-12-24 03:25:16,735 - INFO - New Best Test Accuracy: 80.17%
2024-12-24 03:25:16,735 - INFO - 
Epoch 18/150
2024-12-24 03:25:42,971 - INFO - Training Accuracy: 75.14%
2024-12-24 03:25:45,040 - INFO - Test Accuracy: 80.23%
2024-12-24 03:25:45,047 - INFO - New Best Test Accuracy: 80.23%
2024-12-24 03:25:45,047 - INFO - 
Epoch 19/150
2024-12-24 03:26:08,777 - INFO - Training Accuracy: 75.47%
2024-12-24 03:26:10,826 - INFO - Test Accuracy: 80.27%
2024-12-24 03:26:10,832 - INFO - New Best Test Accuracy: 80.27%
2024-12-24 03:26:10,833 - INFO - 
Epoch 20/150
2024-12-24 03:26:39,536 - INFO - Training Accuracy: 76.02%
2024-12-24 03:26:41,469 - INFO - Test Accuracy: 80.58%
2024-12-24 03:26:41,476 - INFO - New Best Test Accuracy: 80.58%
2024-12-24 03:26:41,476 - INFO - 
Epoch 21/150
2024-12-24 03:27:11,340 - INFO - Training Accuracy: 76.28%
2024-12-24 03:27:13,411 - INFO - Test Accuracy: 80.25%
2024-12-24 03:27:13,412 - INFO - 
Epoch 22/150
2024-12-24 03:27:43,596 - INFO - Training Accuracy: 76.63%
2024-12-24 03:27:45,620 - INFO - Test Accuracy: 82.21%
2024-12-24 03:27:45,627 - INFO - New Best Test Accuracy: 82.21%
2024-12-24 03:27:45,627 - INFO - 
Epoch 23/150
2024-12-24 03:28:15,899 - INFO - Training Accuracy: 76.90%
2024-12-24 03:28:17,872 - INFO - Test Accuracy: 81.72%
2024-12-24 03:28:17,872 - INFO - 
Epoch 24/150
2024-12-24 03:28:46,695 - INFO - Training Accuracy: 77.18%
2024-12-24 03:28:48,987 - INFO - Test Accuracy: 81.67%
2024-12-24 03:28:48,987 - INFO - 
Epoch 25/150
2024-12-24 03:29:19,597 - INFO - Training Accuracy: 77.36%
2024-12-24 03:29:21,669 - INFO - Test Accuracy: 82.19%
2024-12-24 03:29:21,669 - INFO - 
Epoch 26/150
2024-12-24 03:29:52,944 - INFO - Training Accuracy: 77.36%
2024-12-24 03:29:55,098 - INFO - Test Accuracy: 82.20%
2024-12-24 03:29:55,098 - INFO - 
Epoch 27/150
2024-12-24 03:30:25,274 - INFO - Training Accuracy: 79.79%
2024-12-24 03:30:27,289 - INFO - Test Accuracy: 83.58%
2024-12-24 03:30:27,297 - INFO - New Best Test Accuracy: 83.58%
2024-12-24 03:30:27,297 - INFO - 
Epoch 28/150
2024-12-24 03:30:55,747 - INFO - Training Accuracy: 80.79%
2024-12-24 03:30:57,820 - INFO - Test Accuracy: 83.50%
2024-12-24 03:30:57,820 - INFO - 
Epoch 29/150
2024-12-24 03:31:27,429 - INFO - Training Accuracy: 80.64%
2024-12-24 03:31:29,496 - INFO - Test Accuracy: 83.72%
2024-12-24 03:31:29,503 - INFO - New Best Test Accuracy: 83.72%
2024-12-24 03:31:29,503 - INFO - 
Epoch 30/150
2024-12-24 03:32:00,013 - INFO - Training Accuracy: 80.70%
2024-12-24 03:32:02,142 - INFO - Test Accuracy: 83.98%
2024-12-24 03:32:02,149 - INFO - New Best Test Accuracy: 83.98%
2024-12-24 03:32:02,149 - INFO - 
Epoch 31/150
2024-12-24 03:32:22,394 - INFO - Training Accuracy: 81.33%
2024-12-24 03:32:24,323 - INFO - Test Accuracy: 83.75%
2024-12-24 03:32:24,323 - INFO - 
Epoch 32/150
2024-12-24 03:32:53,775 - INFO - Training Accuracy: 81.61%
2024-12-24 03:32:55,910 - INFO - Test Accuracy: 83.91%
2024-12-24 03:32:55,911 - INFO - 
Epoch 33/150
2024-12-24 03:33:25,978 - INFO - Training Accuracy: 81.31%
2024-12-24 03:33:27,908 - INFO - Test Accuracy: 84.04%
2024-12-24 03:33:27,916 - INFO - New Best Test Accuracy: 84.04%
2024-12-24 03:33:27,916 - INFO - 
Epoch 34/150
2024-12-24 03:33:57,441 - INFO - Training Accuracy: 81.37%
2024-12-24 03:33:59,479 - INFO - Test Accuracy: 83.83%
2024-12-24 03:33:59,479 - INFO - 
Epoch 35/150
2024-12-24 03:34:29,617 - INFO - Training Accuracy: 81.70%
2024-12-24 03:34:31,584 - INFO - Test Accuracy: 84.22%
2024-12-24 03:34:31,591 - INFO - New Best Test Accuracy: 84.22%
2024-12-24 03:34:31,591 - INFO - 
Epoch 36/150
2024-12-24 03:35:01,041 - INFO - Training Accuracy: 81.80%
2024-12-24 03:35:02,942 - INFO - Test Accuracy: 84.70%
2024-12-24 03:35:02,949 - INFO - New Best Test Accuracy: 84.70%
2024-12-24 03:35:02,949 - INFO - 
Epoch 37/150
2024-12-24 03:35:31,335 - INFO - Training Accuracy: 81.84%
2024-12-24 03:35:33,336 - INFO - Test Accuracy: 84.00%
2024-12-24 03:35:33,336 - INFO - 
Epoch 38/150
2024-12-24 03:36:01,899 - INFO - Training Accuracy: 81.90%
2024-12-24 03:36:04,279 - INFO - Test Accuracy: 84.55%
2024-12-24 03:36:04,279 - INFO - 
Epoch 39/150
2024-12-24 03:36:34,202 - INFO - Training Accuracy: 81.91%
2024-12-24 03:36:36,269 - INFO - Test Accuracy: 84.45%
2024-12-24 03:36:36,269 - INFO - 
Epoch 40/150
2024-12-24 03:37:06,487 - INFO - Training Accuracy: 81.96%
2024-12-24 03:37:08,750 - INFO - Test Accuracy: 84.43%
2024-12-24 03:37:08,750 - INFO - 
Epoch 41/150
2024-12-24 03:37:39,200 - INFO - Training Accuracy: 82.54%
2024-12-24 03:37:41,262 - INFO - Test Accuracy: 84.56%
2024-12-24 03:37:41,262 - INFO - 
Epoch 42/150
2024-12-24 03:38:11,689 - INFO - Training Accuracy: 82.73%
2024-12-24 03:38:13,753 - INFO - Test Accuracy: 84.66%
2024-12-24 03:38:13,753 - INFO - 
Epoch 43/150
2024-12-24 03:38:43,746 - INFO - Training Accuracy: 82.97%
2024-12-24 03:38:45,795 - INFO - Test Accuracy: 84.68%
2024-12-24 03:38:45,795 - INFO - 
Epoch 44/150
2024-12-24 03:39:15,913 - INFO - Training Accuracy: 82.82%
2024-12-24 03:39:17,948 - INFO - Test Accuracy: 84.65%
2024-12-24 03:39:17,948 - INFO - 
Epoch 45/150
2024-12-24 03:39:48,373 - INFO - Training Accuracy: 82.79%
2024-12-24 03:39:50,382 - INFO - Test Accuracy: 84.54%
2024-12-24 03:39:50,382 - INFO - 
Epoch 46/150
2024-12-24 03:40:20,503 - INFO - Training Accuracy: 82.68%
2024-12-24 03:40:22,462 - INFO - Test Accuracy: 84.80%
2024-12-24 03:40:22,469 - INFO - New Best Test Accuracy: 84.80%
2024-12-24 03:40:22,469 - INFO - 
Epoch 47/150
2024-12-24 03:40:52,757 - INFO - Training Accuracy: 82.78%
2024-12-24 03:40:54,758 - INFO - Test Accuracy: 84.85%
2024-12-24 03:40:54,765 - INFO - New Best Test Accuracy: 84.85%
2024-12-24 03:40:54,765 - INFO - 
Epoch 48/150
2024-12-24 03:41:22,956 - INFO - Training Accuracy: 82.96%
2024-12-24 03:41:25,153 - INFO - Test Accuracy: 84.82%
2024-12-24 03:41:25,153 - INFO - 
Epoch 49/150
2024-12-24 03:41:56,369 - INFO - Training Accuracy: 82.83%
2024-12-24 03:41:58,607 - INFO - Test Accuracy: 84.99%
2024-12-24 03:41:58,615 - INFO - New Best Test Accuracy: 84.99%
2024-12-24 03:41:58,615 - INFO - 
Epoch 50/150
2024-12-24 03:42:30,115 - INFO - Training Accuracy: 82.80%
2024-12-24 03:42:32,426 - INFO - Test Accuracy: 84.85%
2024-12-24 03:42:32,426 - INFO - 
Epoch 51/150
2024-12-24 03:43:01,344 - INFO - Training Accuracy: 82.82%
2024-12-24 03:43:03,339 - INFO - Test Accuracy: 84.76%
2024-12-24 03:43:03,339 - INFO - 
Epoch 52/150
2024-12-24 03:43:33,735 - INFO - Training Accuracy: 82.96%
2024-12-24 03:43:35,696 - INFO - Test Accuracy: 84.87%
2024-12-24 03:43:35,696 - INFO - 
Epoch 53/150
2024-12-24 03:44:06,589 - INFO - Training Accuracy: 83.13%
2024-12-24 03:44:08,494 - INFO - Test Accuracy: 84.96%
2024-12-24 03:44:08,494 - INFO - 
Epoch 54/150
2024-12-24 03:44:38,783 - INFO - Training Accuracy: 83.17%
2024-12-24 03:44:40,752 - INFO - Test Accuracy: 84.93%
2024-12-24 03:44:40,752 - INFO - 
Epoch 55/150
2024-12-24 03:45:11,276 - INFO - Training Accuracy: 83.20%
2024-12-24 03:45:13,570 - INFO - Test Accuracy: 84.98%
2024-12-24 03:45:13,570 - INFO - 
Epoch 56/150
2024-12-24 03:45:44,124 - INFO - Training Accuracy: 83.19%
2024-12-24 03:45:46,337 - INFO - Test Accuracy: 84.94%
2024-12-24 03:45:46,337 - INFO - 
Epoch 57/150
2024-12-24 03:46:17,010 - INFO - Training Accuracy: 82.93%
2024-12-24 03:46:19,021 - INFO - Test Accuracy: 85.04%
2024-12-24 03:46:19,028 - INFO - New Best Test Accuracy: 85.04%
2024-12-24 03:46:19,028 - INFO - 
Epoch 58/150
2024-12-24 03:46:46,476 - INFO - Training Accuracy: 83.32%
2024-12-24 03:46:48,490 - INFO - Test Accuracy: 85.03%
2024-12-24 03:46:48,491 - INFO - 
Epoch 59/150
2024-12-24 03:47:18,444 - INFO - Training Accuracy: 83.46%
2024-12-24 03:47:20,448 - INFO - Test Accuracy: 84.95%
2024-12-24 03:47:20,448 - INFO - 
Epoch 60/150
2024-12-24 03:47:50,463 - INFO - Training Accuracy: 83.25%
2024-12-24 03:47:52,408 - INFO - Test Accuracy: 84.94%
2024-12-24 03:47:52,408 - INFO - 
Epoch 61/150
2024-12-24 03:48:16,191 - INFO - Training Accuracy: 83.11%
2024-12-24 03:48:18,375 - INFO - Test Accuracy: 85.07%
2024-12-24 03:48:18,382 - INFO - New Best Test Accuracy: 85.07%
2024-12-24 03:48:18,382 - INFO - 
Epoch 62/150
2024-12-24 03:48:45,184 - INFO - Training Accuracy: 83.27%
2024-12-24 03:48:47,378 - INFO - Test Accuracy: 85.02%
2024-12-24 03:48:47,379 - INFO - 
Epoch 63/150
2024-12-24 03:49:17,796 - INFO - Training Accuracy: 83.10%
2024-12-24 03:49:19,780 - INFO - Test Accuracy: 84.94%
2024-12-24 03:49:19,781 - INFO - 
Epoch 64/150
2024-12-24 03:49:49,065 - INFO - Training Accuracy: 83.51%
2024-12-24 03:49:51,061 - INFO - Test Accuracy: 84.96%
2024-12-24 03:49:51,061 - INFO - 
Epoch 65/150
2024-12-24 03:50:20,821 - INFO - Training Accuracy: 83.29%
2024-12-24 03:50:22,782 - INFO - Test Accuracy: 84.97%
2024-12-24 03:50:22,782 - INFO - 
Epoch 66/150
2024-12-24 03:50:43,181 - INFO - Training Accuracy: 83.33%
2024-12-24 03:50:45,214 - INFO - Test Accuracy: 84.89%
2024-12-24 03:50:45,214 - INFO - 
Epoch 67/150
2024-12-24 03:51:11,991 - INFO - Training Accuracy: 83.11%
2024-12-24 03:51:13,973 - INFO - Test Accuracy: 84.92%
2024-12-24 03:51:13,973 - INFO - 
Epoch 68/150
2024-12-24 03:51:43,394 - INFO - Training Accuracy: 83.28%
2024-12-24 03:51:45,397 - INFO - Test Accuracy: 84.98%
2024-12-24 03:51:45,397 - INFO - 
Epoch 69/150
2024-12-24 03:52:15,309 - INFO - Training Accuracy: 83.45%
2024-12-24 03:52:17,451 - INFO - Test Accuracy: 84.87%
2024-12-24 03:52:17,451 - INFO - 
Epoch 70/150
2024-12-24 03:52:47,500 - INFO - Training Accuracy: 83.52%
2024-12-24 03:52:49,550 - INFO - Test Accuracy: 84.99%
2024-12-24 03:52:49,550 - INFO - 
Epoch 71/150
2024-12-24 03:53:19,854 - INFO - Training Accuracy: 83.24%
2024-12-24 03:53:21,849 - INFO - Test Accuracy: 84.94%
2024-12-24 03:53:21,849 - INFO - 
Epoch 72/150
2024-12-24 03:53:51,332 - INFO - Training Accuracy: 83.41%
2024-12-24 03:53:53,359 - INFO - Test Accuracy: 85.01%
2024-12-24 03:53:53,359 - INFO - 
Epoch 73/150
2024-12-24 03:54:23,662 - INFO - Training Accuracy: 83.20%
2024-12-24 03:54:25,618 - INFO - Test Accuracy: 84.92%
2024-12-24 03:54:25,618 - INFO - 
Epoch 74/150
2024-12-24 03:54:53,967 - INFO - Training Accuracy: 83.28%
2024-12-24 03:54:55,982 - INFO - Test Accuracy: 84.92%
2024-12-24 03:54:55,982 - INFO - 
Epoch 75/150
2024-12-24 03:55:16,181 - INFO - Training Accuracy: 83.50%
2024-12-24 03:55:18,199 - INFO - Test Accuracy: 84.92%
2024-12-24 03:55:18,199 - INFO - 
Epoch 76/150
2024-12-24 03:55:38,289 - INFO - Training Accuracy: 83.44%
2024-12-24 03:55:40,286 - INFO - Test Accuracy: 84.88%
2024-12-24 03:55:40,286 - INFO - 
Epoch 77/150
2024-12-24 03:56:07,639 - INFO - Training Accuracy: 83.28%
2024-12-24 03:56:09,536 - INFO - Test Accuracy: 84.99%
2024-12-24 03:56:09,536 - INFO - 
Epoch 78/150
2024-12-24 03:56:30,924 - INFO - Training Accuracy: 83.26%
2024-12-24 03:56:33,090 - INFO - Test Accuracy: 85.04%
2024-12-24 03:56:33,090 - INFO - 
Epoch 79/150
2024-12-24 03:57:02,050 - INFO - Training Accuracy: 83.36%
2024-12-24 03:57:04,052 - INFO - Test Accuracy: 85.02%
2024-12-24 03:57:04,052 - INFO - 
Epoch 80/150
2024-12-24 03:57:33,667 - INFO - Training Accuracy: 83.42%
2024-12-24 03:57:35,639 - INFO - Test Accuracy: 85.05%
2024-12-24 03:57:35,639 - INFO - 
Epoch 81/150
2024-12-24 03:58:05,290 - INFO - Training Accuracy: 83.30%
2024-12-24 03:58:07,416 - INFO - Test Accuracy: 85.01%
2024-12-24 03:58:07,416 - INFO - 
Epoch 82/150
2024-12-24 03:58:37,181 - INFO - Training Accuracy: 83.40%
2024-12-24 03:58:39,229 - INFO - Test Accuracy: 84.99%
2024-12-24 03:58:39,229 - INFO - 
Epoch 83/150
2024-12-24 03:59:08,888 - INFO - Training Accuracy: 83.46%
2024-12-24 03:59:10,880 - INFO - Test Accuracy: 85.02%
2024-12-24 03:59:10,880 - INFO - 
Epoch 84/150
2024-12-24 03:59:42,302 - INFO - Training Accuracy: 83.34%
2024-12-24 03:59:44,318 - INFO - Test Accuracy: 84.94%
2024-12-24 03:59:44,318 - INFO - 
Epoch 85/150
2024-12-24 04:00:08,822 - INFO - Training Accuracy: 83.33%
2024-12-24 04:00:10,830 - INFO - Test Accuracy: 85.06%
2024-12-24 04:00:10,830 - INFO - 
Epoch 86/150
2024-12-24 04:00:41,227 - INFO - Training Accuracy: 83.73%
2024-12-24 04:00:43,315 - INFO - Test Accuracy: 84.98%
2024-12-24 04:00:43,315 - INFO - 
Epoch 87/150
2024-12-24 04:01:13,285 - INFO - Training Accuracy: 83.29%
2024-12-24 04:01:15,290 - INFO - Test Accuracy: 85.09%
2024-12-24 04:01:15,297 - INFO - New Best Test Accuracy: 85.09%
2024-12-24 04:01:15,297 - INFO - 
Epoch 88/150
2024-12-24 04:01:44,996 - INFO - Training Accuracy: 83.60%
2024-12-24 04:01:46,959 - INFO - Test Accuracy: 84.90%
2024-12-24 04:01:46,959 - INFO - 
Epoch 89/150
2024-12-24 04:02:13,267 - INFO - Training Accuracy: 83.47%
2024-12-24 04:02:15,270 - INFO - Test Accuracy: 84.92%
2024-12-24 04:02:15,270 - INFO - 
Epoch 90/150
2024-12-24 04:02:44,367 - INFO - Training Accuracy: 83.21%
2024-12-24 04:02:46,606 - INFO - Test Accuracy: 85.13%
2024-12-24 04:02:46,614 - INFO - New Best Test Accuracy: 85.13%
2024-12-24 04:02:46,614 - INFO - 
Epoch 91/150
2024-12-24 04:03:14,533 - INFO - Training Accuracy: 83.34%
2024-12-24 04:03:16,519 - INFO - Test Accuracy: 84.95%
2024-12-24 04:03:16,520 - INFO - 
Epoch 92/150
2024-12-24 04:03:37,037 - INFO - Training Accuracy: 83.40%
2024-12-24 04:03:39,053 - INFO - Test Accuracy: 85.06%
2024-12-24 04:03:39,054 - INFO - 
Epoch 93/150
2024-12-24 04:04:10,188 - INFO - Training Accuracy: 83.35%
2024-12-24 04:04:12,274 - INFO - Test Accuracy: 85.06%
2024-12-24 04:04:12,274 - INFO - 
Epoch 94/150
2024-12-24 04:04:44,450 - INFO - Training Accuracy: 83.57%
2024-12-24 04:04:46,659 - INFO - Test Accuracy: 85.13%
2024-12-24 04:04:46,659 - INFO - 
Epoch 95/150
2024-12-24 04:05:17,553 - INFO - Training Accuracy: 83.41%
2024-12-24 04:05:19,539 - INFO - Test Accuracy: 84.97%
2024-12-24 04:05:19,539 - INFO - 
Epoch 96/150
2024-12-24 04:05:49,622 - INFO - Training Accuracy: 83.25%
2024-12-24 04:05:51,639 - INFO - Test Accuracy: 85.01%
2024-12-24 04:05:51,639 - INFO - 
Epoch 97/150
2024-12-24 04:06:21,817 - INFO - Training Accuracy: 83.20%
2024-12-24 04:06:23,830 - INFO - Test Accuracy: 84.93%
2024-12-24 04:06:23,830 - INFO - 
Epoch 98/150
2024-12-24 04:06:50,869 - INFO - Training Accuracy: 83.32%
2024-12-24 04:06:52,808 - INFO - Test Accuracy: 84.91%
2024-12-24 04:06:52,808 - INFO - 
Epoch 99/150
2024-12-24 04:07:23,248 - INFO - Training Accuracy: 83.07%
2024-12-24 04:07:25,333 - INFO - Test Accuracy: 84.89%
2024-12-24 04:07:25,333 - INFO - 
Epoch 100/150
2024-12-24 04:07:55,641 - INFO - Training Accuracy: 83.37%
2024-12-24 04:07:57,752 - INFO - Test Accuracy: 84.99%
2024-12-24 04:07:57,752 - INFO - 
Epoch 101/150
2024-12-24 04:08:28,546 - INFO - Training Accuracy: 83.49%
2024-12-24 04:08:30,800 - INFO - Test Accuracy: 84.92%
2024-12-24 04:08:30,800 - INFO - 
Epoch 102/150
2024-12-24 04:09:01,665 - INFO - Training Accuracy: 83.43%
2024-12-24 04:09:03,616 - INFO - Test Accuracy: 85.09%
2024-12-24 04:09:03,616 - INFO - 
Epoch 103/150
2024-12-24 04:09:34,081 - INFO - Training Accuracy: 83.28%
2024-12-24 04:09:36,008 - INFO - Test Accuracy: 85.08%
2024-12-24 04:09:36,009 - INFO - 
Epoch 104/150
2024-12-24 04:10:06,258 - INFO - Training Accuracy: 83.49%
2024-12-24 04:10:08,246 - INFO - Test Accuracy: 85.05%
2024-12-24 04:10:08,246 - INFO - 
Epoch 105/150
2024-12-24 04:10:36,083 - INFO - Training Accuracy: 83.53%
2024-12-24 04:10:38,152 - INFO - Test Accuracy: 84.98%
2024-12-24 04:10:38,152 - INFO - 
Epoch 106/150
2024-12-24 04:11:07,013 - INFO - Training Accuracy: 83.18%
2024-12-24 04:11:09,023 - INFO - Test Accuracy: 84.93%
2024-12-24 04:11:09,023 - INFO - 
Epoch 107/150
2024-12-24 04:11:35,674 - INFO - Training Accuracy: 83.48%
2024-12-24 04:11:37,659 - INFO - Test Accuracy: 85.11%
2024-12-24 04:11:37,660 - INFO - 
Epoch 108/150
2024-12-24 04:12:05,292 - INFO - Training Accuracy: 83.39%
2024-12-24 04:12:07,245 - INFO - Test Accuracy: 84.94%
2024-12-24 04:12:07,246 - INFO - 
Epoch 109/150
2024-12-24 04:12:37,665 - INFO - Training Accuracy: 83.35%
2024-12-24 04:12:39,716 - INFO - Test Accuracy: 84.98%
2024-12-24 04:12:39,717 - INFO - 
Epoch 110/150
2024-12-24 04:13:10,017 - INFO - Training Accuracy: 83.46%
2024-12-24 04:13:12,045 - INFO - Test Accuracy: 84.93%
2024-12-24 04:13:12,045 - INFO - 
Epoch 111/150
2024-12-24 04:13:42,135 - INFO - Training Accuracy: 83.68%
2024-12-24 04:13:44,224 - INFO - Test Accuracy: 85.08%
2024-12-24 04:13:44,224 - INFO - 
Epoch 112/150
2024-12-24 04:14:14,811 - INFO - Training Accuracy: 83.59%
2024-12-24 04:14:16,815 - INFO - Test Accuracy: 85.01%
2024-12-24 04:14:16,815 - INFO - 
Epoch 113/150
2024-12-24 04:14:47,146 - INFO - Training Accuracy: 83.55%
2024-12-24 04:14:49,114 - INFO - Test Accuracy: 84.96%
2024-12-24 04:14:49,114 - INFO - 
Epoch 114/150
2024-12-24 04:15:19,481 - INFO - Training Accuracy: 83.42%
2024-12-24 04:15:21,443 - INFO - Test Accuracy: 85.02%
2024-12-24 04:15:21,443 - INFO - 
Epoch 115/150
2024-12-24 04:15:51,050 - INFO - Training Accuracy: 83.33%
2024-12-24 04:15:53,136 - INFO - Test Accuracy: 85.02%
2024-12-24 04:15:53,136 - INFO - 
Epoch 116/150
2024-12-24 04:16:23,385 - INFO - Training Accuracy: 83.55%
2024-12-24 04:16:25,476 - INFO - Test Accuracy: 85.08%
2024-12-24 04:16:25,476 - INFO - 
Epoch 117/150
2024-12-24 04:16:54,022 - INFO - Training Accuracy: 83.63%
2024-12-24 04:16:56,052 - INFO - Test Accuracy: 84.96%
2024-12-24 04:16:56,052 - INFO - 
Epoch 118/150
2024-12-24 04:17:26,066 - INFO - Training Accuracy: 83.50%
2024-12-24 04:17:28,070 - INFO - Test Accuracy: 85.02%
2024-12-24 04:17:28,070 - INFO - 
Epoch 119/150
2024-12-24 04:17:58,277 - INFO - Training Accuracy: 83.40%
2024-12-24 04:18:00,361 - INFO - Test Accuracy: 85.02%
2024-12-24 04:18:00,362 - INFO - 
Epoch 120/150
2024-12-24 04:18:30,671 - INFO - Training Accuracy: 83.56%
2024-12-24 04:18:32,687 - INFO - Test Accuracy: 84.92%
2024-12-24 04:18:32,687 - INFO - 
Epoch 121/150
2024-12-24 04:19:02,855 - INFO - Training Accuracy: 83.30%
2024-12-24 04:19:04,974 - INFO - Test Accuracy: 85.02%
2024-12-24 04:19:04,974 - INFO - 
Epoch 122/150
2024-12-24 04:19:35,273 - INFO - Training Accuracy: 83.53%
2024-12-24 04:19:37,289 - INFO - Test Accuracy: 84.99%
2024-12-24 04:19:37,289 - INFO - 
Epoch 123/150
2024-12-24 04:20:07,905 - INFO - Training Accuracy: 83.61%
2024-12-24 04:20:09,830 - INFO - Test Accuracy: 85.09%
2024-12-24 04:20:09,830 - INFO - 
Epoch 124/150
2024-12-24 04:20:40,304 - INFO - Training Accuracy: 83.65%
2024-12-24 04:20:42,349 - INFO - Test Accuracy: 85.01%
2024-12-24 04:20:42,349 - INFO - 
Epoch 125/150
2024-12-24 04:21:12,354 - INFO - Training Accuracy: 83.79%
2024-12-24 04:21:14,609 - INFO - Test Accuracy: 84.97%
2024-12-24 04:21:14,610 - INFO - 
Epoch 126/150
2024-12-24 04:21:45,614 - INFO - Training Accuracy: 83.59%
2024-12-24 04:21:47,645 - INFO - Test Accuracy: 85.06%
2024-12-24 04:21:47,645 - INFO - 
Epoch 127/150
2024-12-24 04:22:16,445 - INFO - Training Accuracy: 83.46%
2024-12-24 04:22:18,435 - INFO - Test Accuracy: 85.07%
2024-12-24 04:22:18,435 - INFO - 
Epoch 128/150
2024-12-24 04:22:47,905 - INFO - Training Accuracy: 83.63%
2024-12-24 04:22:49,984 - INFO - Test Accuracy: 84.99%
2024-12-24 04:22:49,984 - INFO - 
Epoch 129/150
2024-12-24 04:23:20,480 - INFO - Training Accuracy: 83.58%
2024-12-24 04:23:22,390 - INFO - Test Accuracy: 84.98%
2024-12-24 04:23:22,390 - INFO - 
Epoch 130/150
2024-12-24 04:23:52,060 - INFO - Training Accuracy: 83.40%
2024-12-24 04:23:54,179 - INFO - Test Accuracy: 85.02%
2024-12-24 04:23:54,179 - INFO - 
Epoch 131/150
2024-12-24 04:24:24,596 - INFO - Training Accuracy: 83.55%
2024-12-24 04:24:26,591 - INFO - Test Accuracy: 85.04%
2024-12-24 04:24:26,591 - INFO - 
Epoch 132/150
2024-12-24 04:24:55,928 - INFO - Training Accuracy: 83.63%
2024-12-24 04:24:58,048 - INFO - Test Accuracy: 85.06%
2024-12-24 04:24:58,048 - INFO - 
Epoch 133/150
2024-12-24 04:25:28,239 - INFO - Training Accuracy: 83.49%
2024-12-24 04:25:30,254 - INFO - Test Accuracy: 84.88%
2024-12-24 04:25:30,254 - INFO - 
Epoch 134/150
2024-12-24 04:25:56,282 - INFO - Training Accuracy: 83.55%
2024-12-24 04:25:58,334 - INFO - Test Accuracy: 85.10%
2024-12-24 04:25:58,334 - INFO - 
Epoch 135/150
2024-12-24 04:26:21,187 - INFO - Training Accuracy: 83.40%
2024-12-24 04:26:23,273 - INFO - Test Accuracy: 84.97%
2024-12-24 04:26:23,273 - INFO - 
Epoch 136/150
2024-12-24 04:26:44,519 - INFO - Training Accuracy: 83.54%
2024-12-24 04:26:46,451 - INFO - Test Accuracy: 85.06%
2024-12-24 04:26:46,451 - INFO - 
Epoch 137/150
2024-12-24 04:27:16,448 - INFO - Training Accuracy: 83.83%
2024-12-24 04:27:18,411 - INFO - Test Accuracy: 85.08%
2024-12-24 04:27:18,411 - INFO - 
Epoch 138/150
2024-12-24 04:27:48,491 - INFO - Training Accuracy: 83.61%
2024-12-24 04:27:50,456 - INFO - Test Accuracy: 85.11%
2024-12-24 04:27:50,457 - INFO - 
Epoch 139/150
2024-12-24 04:28:20,711 - INFO - Training Accuracy: 83.67%
2024-12-24 04:28:22,796 - INFO - Test Accuracy: 84.91%
2024-12-24 04:28:22,796 - INFO - 
Epoch 140/150
2024-12-24 04:28:52,929 - INFO - Training Accuracy: 83.60%
2024-12-24 04:28:54,989 - INFO - Test Accuracy: 85.07%
2024-12-24 04:28:54,990 - INFO - 
Epoch 141/150
2024-12-24 04:29:25,107 - INFO - Training Accuracy: 83.49%
2024-12-24 04:29:27,172 - INFO - Test Accuracy: 85.06%
2024-12-24 04:29:27,172 - INFO - 
Epoch 142/150
2024-12-24 04:29:57,353 - INFO - Training Accuracy: 83.71%
2024-12-24 04:29:59,383 - INFO - Test Accuracy: 84.90%
2024-12-24 04:29:59,383 - INFO - 
Epoch 143/150
2024-12-24 04:30:29,386 - INFO - Training Accuracy: 83.51%
2024-12-24 04:30:31,464 - INFO - Test Accuracy: 85.25%
2024-12-24 04:30:31,471 - INFO - New Best Test Accuracy: 85.25%
2024-12-24 04:30:31,472 - INFO - 
Epoch 144/150
2024-12-24 04:31:01,190 - INFO - Training Accuracy: 83.64%
2024-12-24 04:31:03,244 - INFO - Test Accuracy: 85.07%
2024-12-24 04:31:03,244 - INFO - 
Epoch 145/150
2024-12-24 04:31:30,199 - INFO - Training Accuracy: 83.77%
2024-12-24 04:31:32,287 - INFO - Test Accuracy: 85.01%
2024-12-24 04:31:32,287 - INFO - 
Epoch 146/150
2024-12-24 04:32:01,249 - INFO - Training Accuracy: 83.55%
2024-12-24 04:32:03,293 - INFO - Test Accuracy: 85.02%
2024-12-24 04:32:03,293 - INFO - 
Epoch 147/150
2024-12-24 04:32:35,323 - INFO - Training Accuracy: 83.73%
2024-12-24 04:32:37,266 - INFO - Test Accuracy: 84.94%
2024-12-24 04:32:37,266 - INFO - 
Epoch 148/150
2024-12-24 04:33:06,857 - INFO - Training Accuracy: 83.67%
2024-12-24 04:33:08,882 - INFO - Test Accuracy: 85.08%
2024-12-24 04:33:08,882 - INFO - 
Epoch 149/150
2024-12-24 04:33:39,200 - INFO - Training Accuracy: 83.63%
2024-12-24 04:33:41,289 - INFO - Test Accuracy: 85.09%
2024-12-24 04:33:41,289 - INFO - 
Epoch 150/150
2024-12-24 04:34:10,080 - INFO - Training Accuracy: 83.36%
2024-12-24 04:34:12,150 - INFO - Test Accuracy: 85.04%
2024-12-24 04:34:12,150 - INFO - Training completed!
2024-12-24 04:34:12,150 - INFO - Best Test Accuracy achieved: 85.25%
```
</details>

## 📈 Results
The model achieves:

- ✅ Training accuracy:  **~84%**
- 🎯 Test accuracy: **85.25%**
- 🧮 Total parameters: **178,744**

## 🏅 Model Checkpointing

The best model (based on test accuracy) is automatically saved as best_model.pth. 🏆

