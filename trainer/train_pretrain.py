import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


# def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
#     start_time = time.time()
#     for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
#         input_ids = input_ids.to(args.device)
#         labels = labels.to(args.device)
#         lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr

#         with autocast_ctx:
#             res = model(input_ids, labels=labels)
#             loss = res.loss + res.aux_loss
#             loss = loss / args.accumulation_steps

#         scaler.scale(loss).backward()

#         if step % args.accumulation_steps == 0 or step == iters:
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

#             scaler.step(optimizer)
#             scaler.update()

#             optimizer.zero_grad(set_to_none=True)

#         if step % args.log_interval == 0 or step == iters:
#             spend_time = time.time() - start_time
#             current_loss = loss.item() * args.accumulation_steps
#             current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
#             current_logits_loss = current_loss - current_aux_loss
#             current_lr = optimizer.param_groups[-1]['lr']
#             eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
#             Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
#             if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

#         if (step % args.save_interval == 0 or step == iters) and is_main_process():
#             model.eval()
#             moe_suffix = '_moe' if lm_config.use_moe else ''
#             ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
#             raw_model = model.module if isinstance(model, DistributedDataParallel) else model
#             raw_model = getattr(raw_model, '_orig_mod', raw_model)
#             state_dict = raw_model.state_dict()
#             torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
#             lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
#             model.train()
#             del state_dict

#         del input_ids, labels, res, loss

def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    # 判断分布式进程组是否已经初始化, 单卡启动False, 多卡启动True
    if dist.is_initialized():
        # 多卡同步屏障, 同步不同进程
        dist.barrier()
    # 等待本进程指定GPU上的CUDA工作完成, 
    torch.cuda.synchronize(args.device)
    # 重置PyTorch显存分配器的峰值统计
    torch.cuda.reset_peak_memory_stats(args.device)
    # 训练计时起点
    train_start = time.perf_counter()
    # 创建有效Token计数器, 直接让计数器留在GPU上
    effective_tokens = torch.zeros((), 
                        dtype=torch.long, device=args.device)
    # 统计
    input_tokens_including_padding = 0
    # 记录优化器执行了多少次参数更新
    optimizer_updates = 0
    start_time = time.time()
    # step这里是batch序号, 每次循环batch
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # 统计有效token
        effective_tokens += (labels[:, 1:] != -100).sum()
        # 所有的token数
        input_tokens_including_padding += input_ids.numel()
        
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps
        
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # 获取当前缩放系数
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # 判断优化器有没有被跳过, 因为如果发现无效梯度：跳过这次参数更新
            # 这是混合精度训练中防止无效梯度破坏模型参数的一种保护机制
            if not scaler.is_enabled() or scaler.get_scale() >= scale_before:
                optimizer_updates += 1
    
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, res, loss
        
    # 1.等待当前GPU训练任务全部完成, 计算本轮训练时间（秒）
    torch.cuda.synchronize(args.device)
    elapsed_s = time.perf_counter() - train_start
    # 2.获取本卡峰值显存（字节）
    peak_alloc = torch.cuda.max_memory_allocated(args.device)
    peak_reserved = torch.cuda.max_memory_reserved(args.device)
    # 3.准备Token统计数据
    token_stats = torch.stack([effective_tokens, 
                torch.tensor(input_tokens_including_padding, 
                dtype=torch.long, device=args.device)])
    # 4.本Rank的时间、显存和优化器更新统计
    perf_stats = torch.tensor([elapsed_s, peak_alloc, peak_reserved, optimizer_updates], 
                dtype=torch.float64, device=args.device)  

    # 5.如果是DDP, 则聚合不同GPU的数据
    if dist.is_initialized():
        # Token总数：所有GPU求和
        dist.all_reduce(token_stats, op=dist.ReduceOp.SUM)
        # 时间、峰值显存和更新次数：取各GPU最大值
        dist.all_reduce(perf_stats, op=dist.ReduceOp.MAX)
    # 6.从GPU获取最终统计数据
    total_effective_tokens = int(token_stats[0].item())
    total_input_positions = int(token_stats[1].item())
    elapsed_s, peak_alloc, peak_reserved, updates = (perf_stats.cpu().tolist())
    # 7.计算吞吐率
    effective_tps = total_effective_tokens / max(elapsed_s, 1e-9)
    input_tps = total_input_positions / max(elapsed_s, 1e-9)
    # 8.输出最终实验指标, 仅在主进程打印
    if is_main_process():
        Logger(f"Optimizer step executions: {int(updates)}")
        Logger(f"Effective tokens: {total_effective_tokens}")
        Logger(f"Input positions (incl. padding): {total_input_positions}")
        Logger(f"Effective Tokens/s (global): {effective_tps:.2f}")
        Logger(f"Input Tokens/s (global): {input_tps:.2f}")
        Logger(f"Max per-GPU peak allocated: {peak_alloc / 1024**3:.2f} GiB")
        Logger(f"Max per-GPU peak reserved: {peak_reserved / 1024**3:.2f} GiB")
        Logger(f"Training time: {elapsed_s:.2f} s")
        Logger(f"Training time: {elapsed_s / 60:.2f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--seed', default=42, type=int, help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()