# PP-LCNet 文档图像方向识别 — 微调训练项目

基于 [PaddleX](https://github.com/PaddlePaddle/PaddleX) 框架，对 `PP-LCNet_x1_0_doc_ori` 模型进行二次微调，用于文档图像旋转方向（0°/90°/180°/270°）的四分类识别。

---

## 目录

- [环境准备](#一环境准备)
- [下载官方数据集](#二下载官方数据集)
- [加入自有数据集](#三加入自有数据集)
- [训练模型](#四训练模型)

---

## 一、环境准备

本项目使用 **VS Code Dev Container** 作为统一开发环境，容器内已预装 PaddlePaddle 3.0 及所有依赖，无需手动配置 Python 环境。

### 前置要求

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)（或 Docker Engine）
- [Visual Studio Code](https://code.visualstudio.com/)
- VS Code 扩展：[Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### 启动步骤

1. 用 VS Code 打开本项目根目录：

   ```bash
   code /path/to/this/project
   ```

2. VS Code 检测到 `.devcontainer/devcontainer.json` 后，右下角会弹出提示，点击 **"Reopen in Container"**；或手动按 `F1` 执行命令 **"Dev Containers: Reopen in Container"**。

3. 等待 Docker 镜像构建完成（首次构建约需数分钟，镜像约 5 GB）。构建完成后，VS Code 会自动连接容器，并在容器内自动执行以下初始化命令：
   - 安装 PaddleX 及其 `PaddleOCR`、`PaddleClas` 套件
   - 配置 Git 凭据转发

至此开发环境即可使用，后续所有操作均在容器内的终端中执行。

---

## 二、下载官方数据集

在容器内的终端中执行以下命令，将 PaddleX 官方文档方向数据集下载并解压到 `./dataset` 目录：

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/text_image_orientation.tar \
     -P ./dataset

tar -xf ./dataset/text_image_orientation.tar -C ./dataset/
```

解压后目录结构如下：

```
dataset/
└── text_image_orientation/
    ├── label.txt      # 类别标签（0~3 对应 0°/90°/180°/270°）
    ├── train.txt      # 训练集清单
    ├── val.txt        # 验证集清单
    └── images/        # 图片文件
```

---

## 三、加入自有数据集

如果你有自己的文档图片，可以通过以下步骤将其融入训练集：

### 1. 放入图片

将**正放**（方向正确）的原始图片放入项目根目录下的 `new_images/` 目录：

```
new_images/
├── doc_001.jpg
├── doc_002.png
└── ...
```

> 支持格式：`.jpg` / `.jpeg` / `.png` / `.bmp`

### 2. 运行数据增强脚本

执行 `augment_rotation.py` 脚本。脚本会将每张原始图片旋转为 0°、90°、180°、270° 共 4 个版本，输出到官方数据集的 `images/` 目录，并将对应记录**追加**写入 `train.txt`（已存在的记录自动跳过，不会重复）。

```bash
python augment_rotation.py \
    -i ./new_images \
    -o ./dataset/text_image_orientation/images
```

**参数说明：**

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--input` | `-i` | 原始图片所在目录 | *(必填)* |
| `--output` | `-o` | 增强图片输出目录 | *(必填)* |
| `--train-txt` | `-t` | `train.txt` 文件路径 | `./dataset/text_image_orientation/train.txt` |

脚本执行完毕后会打印处理摘要，例如：

```
找到 10 张图片，开始处理...
[1/10] 处理完成: doc_001.jpg
...
处理完成！
- 新增记录数量: 40
- 跳过重复记录: 0
- 保存位置: ./dataset/text_image_orientation/images
- 标签文件: ./dataset/text_image_orientation/train.txt (追加)
```

---

## 四、训练模型

数据准备完毕后，执行训练脚本：

```bash
bash train.sh
```

训练配置在 [model_config.yaml](model_config.yaml) 中管理，主要参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `Train.epochs_iters` | 训练轮数 | `50` |
| `Train.batch_size` | 批次大小 | `16` |
| `Train.learning_rate` | 学习率 | `0.005` |
| `Global.device` | 训练设备 | `cpu`（可改为 `gpu:0`） |

训练完成后，最优模型权重保存在 `output/best_model/` 目录。
