# miniBuild
动手训练大模型，4张RTX 3060，基于minimand跑通预训练、SFT、LoRA、DPO一直到GRPO的关键训练流程。
- 我的训练配置

| :---:       |    :----:   |     :---:     |        :---:     |
| GPU    | 4 × NVIDIA RTX 3060 |   单卡显存     |      12GB      |
| Header      | Title       | Here's this   |            |
| Paragraph   | Text        | And more      |            |


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






实验配置





训练结果




## 3️⃣ 监督微调 SFT



实验配置





训练结果


