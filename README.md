# DiffSynth_GenWarp

本仓库基于[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/)

原[README](./README_ori_zh.md)

## introduction
利用DiffSynth-Studio仓库，微调qwen-image-edit以适应矫正任务



## 安装
同原仓库

从源码安装（推荐）：

```
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

<details>
<summary>其他安装方式</summary>

从 pypi 安装（存在版本更新延迟，如需使用最新功能，请从源码安装）

```
pip install diffsynth
```

如果在安装过程中遇到问题，可能是由上游依赖包导致的，请参考这些包的文档：

* [torch](https://pytorch.org/get-started/locally/)
* [sentencepiece](https://github.com/google/sentencepiece)
* [cmake](https://cmake.org)
* [cupy](https://docs.cupy.dev/en/stable/install.html)

</details>

### 个人使用的docker:
```bash
registry.intsig.net/yuang_feng/h800_torch2.5.0cu124_diffsynth:v1.0.0
```



## 训练

```
bash scripts/train_sample.sh
```
主要需要修改对应的数据文件路径，其余参数可以沿用默认、

- `csv_path`：指定包含图像路径和元信息的 CSV 文件，用于加载训练数据。
- `train_id`：指定输出目录。


### 📂 数据相关参数

| 参数 | 说明 |
|------|------|
| `--dataset_base_path` | 数据集根目录，自动从 `csv_path` 推导 |
| `--dataset_metadata_path` | 指定包含图像路径和元信息的 CSV 文件 |
| `--data_file_keys` | 指定 CSV 中用于训练的字段，如原图和编辑图 |
| `--extra_inputs` | 除主图像外的额外输入字段 不用改 |
| `--max_pixels` | 限制单张图像的最大像素数（如 1048576） |
| `--dataset_repeat` | 数据集重复次数，重复几次数据集 |

### 🧠 模型与训练参数

| 参数 | 说明 |
|------|------|
| `--learning_rate` | 学习率（如 1e-4） |
| `--num_epochs` | 训练轮数（如 6） |
| `--remove_prefix_in_ckpt` | 加载权重时移除的前缀，不用改（如 `pipe.dit.`） |
| `--output_path` | 训练结果保存路径，这里会根据上面的train_id自动生成 |
| `--save_steps` | 每隔多少步保存一次模型（如 4000） |

### 🔧 LoRA 微调相关参数

| 参数 | 说明 |
|------|------|
| `--lora_base_model` | LoRA 微调的基础模型名称（如 `dit`） |
| `--lora_target_modules` | 需要注入 LoRA 的模块列表 |
| `--lora_rank` | LoRA 的秩（如 32） |

### ⚙️ 训练优化参数

| 参数 | 说明 |
|------|------|
| `--use_gradient_checkpointing` | 启用梯度检查点，节省显存 |
| `--dataset_num_workers` | 数据加载线程数 |
| `--find_unused_parameters` | 启用分布式训练时的参数检查 |

### 🧩 模型组件路径 不改或者改成相应保存路径即可

| 参数 | 说明 |
|------|------|
| `--tokenizer_path` | 文本编码器的 tokenizer 路径 |
| `--processor_path` | 图像处理器路径 |
| `--model_paths` | 模型权重路径列表，包含 transformer、text encoder 和 VAE 的多个分片文件 |



## 推理
```
bash scripts/test_sample.sh
```
请根据需要修改加载的权重，推理的数据，以及prompt



### 基本参数

| 参数 | 说明 |
|------|------|
| `--lora_path` | 指定用于推理的 LoRA 微调模型权重路径（`.safetensors` 文件） |
| `--input_dir` | 输入图像所在目录，支持批量处理 |
| `--output_dir` | 推理结果保存目录，自动包含模型路径、测试集名、图像尺寸与 resize 模式等信息 |
| `--prompt` | 指定图像编辑任务的文本描述，定义矫正目标与保留内容 |
| `--batch_size` | 推理时的batch大小，设为 1 表示逐张处理 |
| `--short_side_pixel` | 图像短边的目标尺寸，影响 resize 行为 |
| `--img_size` | 模型输入图像的最终尺寸（正方形） |
| `--infer_steps` | 推理过程中的迭代步数，影响生成质量与速度 |

---

### 图像预处理参数：`resize_mode`

用于控制推理前图像的尺寸调整方式：

- `crop`：先 resize 使短边符合 `short_side_pixel`，再根据目标 `img_size` 进行方块形状的中心裁剪  
- `stretch`：不裁剪，直接拉伸到 `img_size` 的正方形  
- `scale_to_short_side`：使短边符合 `img_size`，保持长宽比不变  

该参数通过 `--resize_mode=$resize_mode` 传入脚本。

---

### 输出格式说明

推理结果将输出三张图像：

- `a_*.png`：原始输入图像  
- `b_*.png`：矫正后的图像  
- `c_*.png`：原图与矫正图的对比图
输出路径结构示例：

```
{ckpt.visualization}/1020-1_bad2_1024_stretch/
├── a_00001.png
├── b_00001.png
├── c_00001.png
```


<br>
<br>
<br>
<br>
<br>
<br>

