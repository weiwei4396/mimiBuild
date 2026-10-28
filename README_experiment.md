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
|单卡Pretrain|63.91M|768|8|False|None|pretrain_t2t_mini.jsonl|1270238|256|4|2|4|32|混合精度|AdamW|5e-4|Cosine Decay|1|42|1.0|2|

训练过程与性能
|实验|首次Train Loss|最后Train Loss|Optimizer 更新次数|平均有效 Tokens/s（全局）|总训练时间|单卡峰值显存|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|单卡Pretrain|   |   |   |   |   |   |

模型评估



## 3️⃣ 监督微调 SFT

实验配置


训练结果





## 3️⃣ 直接偏好优化 DPO








