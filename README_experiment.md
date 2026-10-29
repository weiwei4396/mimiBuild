# miniBuild
动手训练大模型，4张RTX 3060，基于minimand跑通预训练、SFT、LoRA、DPO一直到GRPO的关键训练流程。
- 我的训练配置

| GPU    | 4 × NVIDIA RTX 3060 |   单卡显存     |      12GB      |
| :-:       |    :-:   |     :-:     |        :-:     |
| PyTorch   |    2.6.0+cu124    | PyTorch CUDA  |    12.4     |
| Python 版本   | 3.10.0  | NVIDIA 驱动版本   |    595.84    |


## 1️⃣ 搭建环境
```shell
git clone https://github.com/jingyaogong/minimind.git
cd minimand
conda create -n mini python=3.10
conda activate mini
pip install -r requirements.txt
git rev-parse HEAD
#f659b55761b754d306bd140573493a6543cafd7f
```


## 2️⃣ 预训练 Pretrain

模型、数据与训练配置
|实验|模型参数量|hidden_size|num_hidden_layers|use_moe|from_weight|数据集|训练样本数|max_seq_len|world_size|每卡batch_size|accumulation_steps|effective_batch_size(执行优化器参数更新时的batch size)|dtype|Optimizer|Learning Rate|学习率调度|Epoch|Seed|grad_clip|num_workers|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|单卡Pretrain|63.91M|768|8|False|None|pretrain_t2t_mini.jsonl|1,270,238|340|1|32|8|32*8=256 ([1,270,238/256]=4,962)|BF16混合精度|AdamW|5e-4|Cosine Decay|2|42|1.0|8|

> --num_workers是PyTorch DataLoader的参数，表示使用多少个子进程并行加载和处理训练数据。在 DDP 多卡训练中，每张 GPU 对应一个训练进程，而每个训练进程都会创建自己的 DataLoader。

> world_size 严格来说表示训练进程数量，不是 GPU 数量。只是我们平常使用 PyTorch DDP 时，通常让一个训练进程负责一张 GPU，所以两者数量相等。

训练过程与性能
|实验|首次Train Loss|最后Train Loss|Optimizer 更新次数|平均有效 Tokens/s（全局）|单卡峰值显存|总训练时间|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|单卡Pretrain|7.5224|1.6552|每个epoch更新4,962次|13,889.16 + 15,062.45|6.12 GiB + 6.11 GiB|317.87 min + 293.11 min|

> 有效token也就是参与Loss计算的token。

> Optimizer 更新次数，每个batch 32条数据，总共[1,270,238/32]=39,695 个 Micro-Batch，每8个Micro-Batch更新一次，[39,695/8]=4962次。

> Allocated (峰值显存) 是 PyTorch 的 Tensor 实际占用的显存；Reserved (显存缓存分配器峰值) 是 PyTorch 显存缓存分配器从 CUDA 获取并管理的显存。

- 实验环境：单张 NVIDIA RTX 3060 12 GiB，使用 MiniMind 的 Pretrain 训练脚本，对 pretrain_t2t_mini.jsonl 中的 1,270,238 条数据训练 2 个 Epoch。
- 训练规模：每个 Epoch 处理 264,899,312 个有效监督Token、431,880,920 个输入位置，并执行 4,962 次优化器更新调用。
- 训练性能：Epoch 1 的有效 Token 吞吐量为 13,889.16 tokens/s，训练耗时 317.87 分钟；Epoch 2 吞吐量提高至 15,062.45 tokens/s，耗时下降至 293.11 分钟。相较于 Epoch 1，第二轮吞吐量提升约 8.45%，训练时间缩短约 7.79%。
- 数据利用情况：有效监督 Token 占输入位置的约 61.34%。结合当前 PretrainDataset 的固定长度 Padding 和 Shift 机制，推算约 38.37% 的输入位置为 Padding，说明后续可以探索减少无效填充的训练数据组织方式。
- 显存表现：两轮 PyTorch Tensor 峰值显存分别为 6.12 GiB 和 6.11 GiB，缓存分配器峰值均为 6.47 GiB，未发现明显的跨 Epoch 显存峰值增长。
- 实验结论：该实验完成了大规模单卡预训练性能基线的采集。两个 Epoch 的统计结果一致性较好，第二轮训练吞吐有所提升，显存表现稳定。




模型评估



## 3️⃣ 监督微调 SFT

实验配置


训练结果





## 3️⃣ 直接偏好优化 DPO








