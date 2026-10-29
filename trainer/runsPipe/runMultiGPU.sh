

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 train_pretrain.py \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --save_weight pt_mini_4gpu \
  --from_weight none \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --epochs 2 \
  --batch_size 32 \
  --accumulation_steps 2 \
  --max_seq_len 340 \
  --learning_rate 5e-4 \
  --dtype bfloat16 \
  --num_workers 2 \
  --log_interval 100 \
  --save_interval 1000 \
  2>&1 | tee ../runs/PT-02/train.log






