# mimiBuild
Training a large language model from scratch

[模型运行记录](https://github.com/weiwei4396/mimiBuild/blob/main/README_experiment.md) | [中文笔记记录](https://ycnaiekpok4f.feishu.cn/wiki/Q36ew8rLCiVKh9kRf2YcLscZnnb?from=from_copylink)


## 数据集

[数据下载地址](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)
- 一般是jsonl形式，json-line形式的json，内存友好，方便分割
- 一些特殊标识符：<BOS>、<EOS>、<PAD>
- pretrain_t2t_mini.jsonl，1270238，1.27 M的序列。
- PretrainDataset 负责把原始文本转换成模型预训练需要的数据。主要看__getitem__()，处理一条样本，生成input_ids和labels。
- SFTDataset的核心任务，就是把原始多轮对话转换成模型训练所需要的input_ids和labels，并通过标签掩码让模型主要学习Assistant的回答 (mask掉其他，只保留Assistant的回答)。
- 

