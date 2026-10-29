# mimiBuild
Training a large language model from scratch

[项目实战记录](https://github.com/weiwei4396/mimiBuild/blob/main/README_experiment.md) | [项目中文笔记记录](https://ycnaiekpok4f.feishu.cn/wiki/Q36ew8rLCiVKh9kRf2YcLscZnnb?from=from_copylink) | [大模型相关笔记](https://ycnaiekpok4f.feishu.cn/wiki/Sm4FwdOvsiFegKkUGFhcZfmwnBd?from=from_copylink)


## 数据集

[数据下载地址](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)
- 一般是jsonl形式，json-line形式的json，内存友好，方便分割
- 一些特殊标识符：<BOS>、<EOS>、<PAD>
- pretrain_t2t_mini.jsonl，1270238，1.27 M的序列。
- PretrainDataset 负责把原始文本转换成模型预训练需要的数据。主要看__getitem__()，处理一条样本，生成input_ids和labels。
- SFTDataset的核心任务，就是把原始多轮对话转换成模型训练所需要的input_ids和labels，并通过标签掩码让模型主要学习Assistant的回答 (mask掉其他，只保留Assistant的回答)。


## 模型组件
- RMSNrom/LayerNorm ✔️
- RoPE/YaRN ✔️
- Attention(GQA) ✔️
- FFN/MOE-FFN ✔️
- Block/Model ✔️


## 预训练
- 关于梯度累积，假设希望使用batch size是32，但是GPU只能容纳batch size为8，这样每处理8个样品，就计算一次梯度，但是暂时不更新参数，accumulation_steps为4，连续处理4个batch之后，把梯度累积起来，再执行一次参数更新。在分布式训练DDP下还需要乘以GPU进程数量。假设有4张GPU，有效batch size为128。

计算梯度、累积梯度、更新模型参数
- loss.backward()：计算梯度，并将梯度累积到模型参数的.grad属性
- optimizer.step()：根据当前累积的梯度和学习率，真正更新模型参数
- optimizer.zero_grad()：清空梯度，为下一轮累积做准备
多个Loss相加后求梯度，等于每个Loss分别求梯度再相加。

GradScaler的主要目的不是减少显存，而是防止反向传播的某些中间计算使用FP16时，过小的梯度因为数值精度限制而变成0 (放大loss，让很小的梯度更容易被FP16表示)。而 unscale_()的作用是：把放大后的梯度恢复到原来的数值，确保后续梯度裁剪和参数更新使用正确的梯度。

> scaler.scale(loss).backward()    <＝==>    scaled_loss = loss * scale; scaled_loss.backward()

> 为了避免丢失梯度，在反向传播之前放大Loss，让反向计算中的梯度也随之放大。等计算完成以后，再把梯度缩小回来，交给优化器更新参数。 
- 关键信息：FP16/BF16主要是为了让GPU在前向传播和反向传播中的部分计算更快、占用更少显存，而不是为了让优化器使用低精度梯度更新参数。

> 哪一步变成FP16/BF16？

> main函数中，autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)，在训练时with autocast_ctx。

> FP32和FP16/BF16混合计算，具体由autocast根据算子选择。PyTorch的官方AMP文档也明确说明，反向传播中的运算会沿用对应前向运算所选择的精度。

在MiniMand中，前向传播和反向传播计算梯度的时候使用混合精度，更新参数和存储参数都是用的全精度。


## 监督微调

- 在做训练的时候，还是会提供给模型整句话(也包括添加的user、Assistant和<|im_start|><|im_end|>等)，只不过，后面预测的时候，损失只包含预测的回答的损失，其他因为都标记为-100，cross entropy忽略了这些损失。
- 监督微调和预训练的差异在于：Dataset如何生成labels，以及模型从什么权重开始训练。监督微调需要mask掉回答之外的token，不计入损失。模型权重，预训练从None开始，监督微调从预训练的权重开始。

## 直接偏好优化 DPO

Next


## 基于人类反馈的强化学习 RLHF

Next





