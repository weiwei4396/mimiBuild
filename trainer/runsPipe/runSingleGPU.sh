
CUDA_VISIBLE_DEVICES=1 python train_pretrain.py \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --save_weight pt_mini \
  --from_weight none \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --epochs 2 \
  --batch_size 32 \
  --accumulation_steps 8 \
  --max_seq_len 340 \
  --learning_rate 5e-4 \
  --dtype bfloat16 \
  --num_workers 8 \
  --log_interval 100 \
  --save_interval 1000


