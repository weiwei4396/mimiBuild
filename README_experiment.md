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
|实验|模型参数量|hidden_size|num_hidden_layers|use_moe|from_weight|数据集|训练样本数|max_seq_len|world_size|每卡batch_size|accumulation_steps|effective_batch_size|dtype|Optimizer|Learning Rate|学习率调度|Epoch|Seed|grad_clip|num_workers|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|单卡Pretrain|63.91M|768|8|False|None|pretrain_t2t_mini.jsonl|1,270,238|340|4|2|4|32|混合精度BF16|AdamW|5e-4|Cosine Decay|2|42|1.0|8|

训练过程与性能
|实验|首次Train Loss|最后Train Loss|Optimizer 更新次数|平均有效 Tokens/s（全局）|单卡峰值显存|总训练时间|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|单卡Pretrain|7.5224|1.6552|每个epoch更新4,962次|13,889.16 + 15,062.45|6.12 GiB + 6.11 GiB|317.87 min + 293.11 min|
- 实验环境：单张NVIDIA RTX 3060 12 GiB，使用MiniMind的Pretrain训练脚本，对pretrain_t2t_mini.jsonl中的1,270,238条数据训练2个Epoch。
- 训练规模：每个Epoch处理264,899,312个有效监督Token、431,880,920个输入位置，并执行4,962次优化器更新调用。
- 训练性能：Epoch 1的有效Token吞吐量为13,889.16 tokens/s，训练耗时317.87分钟；Epoch 2吞吐量提高至15,062.45 tokens/s，耗时下降至293.11分钟。相较于Epoch 1，第二轮吞吐量提升约8.45%，训练时间缩短约7.79%。
- 数据利用情况：有效监督Token占输入位置的约61.34%。结合当前PretrainDataset的固定长度Padding和Shift机制，推算约38.37%的输入位置为Padding，说明后续可以探索减少无效填充的训练数据组织方式。
- 显存表现：两轮PyTorch Tensor峰值显存分别为6.12 GiB和6.11 GiB，缓存分配器峰值均为6.47 GiB，未发现明显的跨Epoch显存峰值增长。
- 实验结论：该实验完成了大规模单卡预训练性能基线的采集。两个Epoch的统计结果一致性较好，第二轮训练吞吐有所提升，显存表现稳定。



模型评估



## 3️⃣ 监督微调 SFT

实验配置


训练结果





## 3️⃣ 直接偏好优化 DPO








