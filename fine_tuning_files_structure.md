# PP-LCNet_x1_0_doc_ori 二次微调相关文件结构图

## 📁 核心配置文件层（用户直接修改）

```
PaddleX/
├── paddlex/
│   └── configs/
│       └── modules/
│           └── doc_text_orientation/
│               └── PP-LCNet_x1_0_doc_ori.yaml  ⭐ [主配置文件]
│                   ├── Global:            # 全局参数
│                   │   ├── model         # 模型名称
│                   │   ├── mode          # 运行模式 (train/evaluate/export/predict)
│                   │   ├── device        # 设备类型 (cpu/gpu)
│                   │   ├── dataset_dir   # 数据集路径
│                   │   └── output        # 输出目录
│                   ├── Train:            # 训练参数 ⚙️
│                   │   ├── num_classes           # 类别数量
│                   │   ├── epochs_iters          # 训练轮数
│                   │   ├── batch_size            # 批次大小
│                   │   ├── learning_rate         # 学习率
│                   │   ├── warmup_steps          # 预热步数
│                   │   ├── pretrain_weight_path  # 预训练权重路径
│                   │   ├── resume_path           # 断点续训路径
│                   │   ├── log_interval          # 日志打印间隔
│                   │   ├── eval_interval         # 验证间隔
│                   │   └── save_interval         # 保存间隔
│                   ├── Evaluate:         # 评估参数
│                   ├── Export:           # 导出参数
│                   └── Predict:          # 预测参数
```

---

## 📁 底层模型配置层（PaddleClas引擎配置）

```
PaddleX/
├── paddlex/
│   └── repo_apis/
│       └── PaddleClas_api/
│           ├── configs/
│           │   └── PP-LCNet_x1_0_doc_ori.yaml  ⭐ [PaddleClas配置]
│           │       ├── Global:                 # PaddleClas全局配置
│           │       │   ├── checkpoints         # 检查点路径
│           │       │   ├── pretrained_model    # 预训练模型
│           │       │   ├── output_dir          # 输出目录
│           │       │   ├── device              # 设备
│           │       │   ├── epochs              # 训练轮数
│           │       │   └── image_shape         # 输入图像尺寸
│           │       ├── AMP:                    # 混合精度训练
│           │       ├── Arch:                   # 模型架构 🏗️
│           │       │   ├── name: PPLCNet_x1_0  # 模型名称
│           │       │   ├── class_num: 4        # 类别数
│           │       │   └── pretrained: True    # 是否使用预训练
│           │       ├── Loss:                   # 损失函数配置 📉
│           │       │   ├── Train
│           │       │   │   └── CELoss:
│           │       │   │       ├── weight: 1.0
│           │       │   │       └── epsilon: 0.1  # 标签平滑参数
│           │       │   └── Eval
│           │       ├── Optimizer:              # 优化器配置 🎯
│           │       │   ├── name: Momentum
│           │       │   ├── momentum: 0.9
│           │       │   ├── lr:                 # 学习率策略
│           │       │   │   ├── name: Cosine
│           │       │   │   ├── learning_rate: 0.4
│           │       │   │   └── warmup_epoch: 5
│           │       │   └── regularizer:        # 正则化
│           │       │       ├── name: 'L2'
│           │       │       └── coeff: 0.00003
│           │       └── DataLoader:             # 数据加载配置 📊
│           │           ├── Train:
│           │           │   ├── dataset:
│           │           │   │   ├── name: ImageNetDataset
│           │           │   │   ├── image_root
│           │           │   │   ├── cls_label_path
│           │           │   │   └── transform_ops:  # 数据增强 🔄
│           │           │   │       ├── DecodeImage
│           │           │   │       ├── RandCropImage
│           │           │   │       ├── TimmAutoAugment
│           │           │   │       ├── NormalizeImage
│           │           │   │       └── RandomErasing
│           │           │   ├── sampler:
│           │           │   │   └── batch_size: 256
│           │           │   └── loader:
│           │           │       └── num_workers: 8
│           │           └── Eval:
│           │               └── dataset:
│           │                   └── transform_ops:
│           │                       ├── DecodeImage
│           │                       ├── ResizeImage
│           │                       ├── CropImage
│           │                       └── NormalizeImage
│           │
│           └── cls/
│               ├── register.py              # 模型注册信息
│               ├── config.py                # 配置类定义
│               ├── model.py                 # 模型封装类
│               └── runner.py                # 训练/评估执行器
```

---

## 📁 模型架构定义层（需要改模型结构时）

```
PaddleX/
└── (依赖外部 PaddleClas 仓库)
    └── paddleclas/
        └── ppcls/
            ├── arch/
            │   └── backbone/
            │       └── legendary_models/
            │           └── pp_lcnet.py  ⭐ [PP-LCNet网络结构定义]
            │               ├── class PPLCNet:
            │               │   ├── __init__():         # 网络初始化
            │               │   │   ├── scale           # 网络宽度缩放 (x1_0表示1.0倍)
            │               │   │   ├── class_num       # 分类数
            │               │   │   ├── dropout_prob    # Dropout概率
            │               │   │   └── use_last_conv   # 是否使用最后的卷积层
            │               │   └── forward():          # 前向传播
            │               └── PPLCNet_x1_0():         # 预定义的x1.0版本
            │
            ├── loss/
            │   └── celoss.py                # 交叉熵损失实现
            │
            ├── optimizer/
            │   └── optimizer.py             # 优化器构建
            │
            └── data/
                ├── dataloader/
                │   └── imagenet_dataset.py  # ImageNet数据集加载器
                └── preprocess/
                    └── ops/
                        └── operators.py     # 数据增强算子实现
```

---

## 📁 训练/评估/导出执行层

```
PaddleX/
├── paddlex/
│   ├── modules/
│   │   └── image_classification/
│   │       ├── trainer.py      ⭐ [训练流程封装]
│   │       ├── evaluator.py       [评估流程封装]
│   │       ├── exportor.py        [导出流程封装]
│   │       └── model_list.py      [支持的模型列表]
│   │
│   └── repo_apis/
│       └── PaddleClas_api/
│           └── cls/
│               └── runner.py   ⭐ [实际执行训练/评估的Runner]
│
└── main.py                     ⭐ [PaddleX统一入口]
```

---

## 📁 数据集目录结构

```
dataset/
└── text_image_orientation/
    ├── train.txt               # 训练集标注文件
    ├── val.txt                 # 验证集标注文件
    ├── label.txt               # 类别标签文件
    └── images/                 # 图像文件夹
        ├── img_0001.jpg
        ├── img_0002.jpg
        └── ...
```

---

## 🎯 二次微调时的修改要点

### 1️⃣ **基础训练参数修改**（最常见）
📄 修改文件：`paddlex/configs/modules/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml`

```yaml
Train:
  num_classes: 4              # 🔧 改为您的类别数
  epochs_iters: 50            # 🔧 调整训练轮数
  batch_size: 16              # 🔧 根据内存/显存调整
  learning_rate: 0.005        # 🔧 调整学习率
  warmup_steps: 5             # 🔧 预热步数
```

### 2️⃣ **高级训练策略修改**（需要更精细控制）
📄 修改文件：`paddlex/repo_apis/PaddleClas_api/configs/PP-LCNet_x1_0_doc_ori.yaml`

```yaml
# 修改优化器
Optimizer:
  name: Momentum              # 🔧 可改为 Adam/AdamW/SGD
  momentum: 0.9
  lr:
    name: Cosine              # 🔧 可改为 Linear/Step/MultiStep
    learning_rate: 0.4
    warmup_epoch: 5           # 🔧 预热轮数
  regularizer:
    coeff: 0.00003            # 🔧 L2正则化系数

# 修改损失函数
Loss:
  Train:
    - CELoss:
        epsilon: 0.1          # 🔧 标签平滑参数 (0.0-0.2)

# 修改数据增强
DataLoader:
  Train:
    dataset:
      transform_ops:
        - RandCropImage:
            size: 224         # 🔧 裁剪尺寸
        - TimmAutoAugment:
            prob: 0.0         # 🔧 自动增强概率 (0.0-1.0)
        - RandomErasing:
            EPSILON: 0.0      # 🔧 随机擦除概率
```

### 3️⃣ **模型架构修改**（需要改网络结构）
📄 修改文件：`paddlex/repo_apis/PaddleClas_api/configs/PP-LCNet_x1_0_doc_ori.yaml`

```yaml
Arch:
  name: PPLCNet_x1_0          # 🔧 可改为 PPLCNet_x0_5/x1_5/x2_0
  class_num: 4                # 🔧 类别数
  pretrained: True
```

如需更深层次修改网络结构（如增加层、改激活函数等），需要修改 PaddleClas 源码中的：
📄 `paddleclas/ppcls/arch/backbone/legendary_models/pp_lcnet.py`

### 4️⃣ **数据集配置**
📄 修改文件：`paddlex/repo_apis/PaddleClas_api/configs/PP-LCNet_x1_0_doc_ori.yaml`

```yaml
DataLoader:
  Train:
    dataset:
      image_root: ./dataset/text_image_orientation/          # 🔧 图像根目录
      cls_label_path: ./dataset/text_image_orientation/train_list.txt  # 🔧 标注文件
  Eval:
    dataset:
      image_root: ./dataset/text_image_orientation/
      cls_label_path: ./dataset/text_image_orientation/test_list.txt
```

---

## 📊 优先级建议

| 修改类型 | 文件位置 | 修改频率 | 难度 |
|---------|---------|---------|------|
| 基础训练参数 | `paddlex/configs/.../PP-LCNet_x1_0_doc_ori.yaml` | ⭐⭐⭐⭐⭐ | ⭐ |
| 数据增强策略 | `paddlex/repo_apis/.../configs/PP-LCNet_x1_0_doc_ori.yaml` | ⭐⭐⭐⭐ | ⭐⭐ |
| 优化器/学习率策略 | `paddlex/repo_apis/.../configs/PP-LCNet_x1_0_doc_ori.yaml` | ⭐⭐⭐ | ⭐⭐ |
| 损失函数 | `paddlex/repo_apis/.../configs/PP-LCNet_x1_0_doc_ori.yaml` | ⭐⭐ | ⭐⭐⭐ |
| 模型架构（scale） | `paddlex/repo_apis/.../configs/PP-LCNet_x1_0_doc_ori.yaml` | ⭐⭐ | ⭐⭐ |
| 网络结构深度修改 | `paddleclas/.../pp_lcnet.py` | ⭐ | ⭐⭐⭐⭐⭐ |

---

## 🚀 快速启动命令

```powershell
# 1. 训练
uv run python main.py `
    -c paddlex/configs/modules/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml `
    -o Global.mode=train `
    -o Global.dataset_dir=E:\Project_Code\paddlex-dev\dataset\text_image_orientation

# 2. 断点续训
uv run python main.py `
    -c paddlex/configs/modules/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml `
    -o Global.mode=train `
    -o Global.dataset_dir=E:\Project_Code\paddlex-dev\dataset\text_image_orientation `
    -o Train.resume_path=output/latest/latest.pdparams

# 3. 导出模型
uv run python main.py `
    -c paddlex/configs/modules/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml `
    -o Global.mode=export `
    -o Export.weight_path=output/best_model/best_model.pdparams

# 4. 评估模型
uv run python main.py `
    -c paddlex/configs/modules/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml `
    -o Global.mode=evaluate `
    -o Evaluate.weight_path=output/best_model/best_model.pdparams
```

---

**生成时间**: 2026年2月1日  
**适用于**: PaddleX PP-LCNet_x1_0_doc_ori 二次微调
**作者**：Claude 
