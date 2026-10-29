
CUDA_VISIBLE_DEVICES=2 python train_pretrain.py \
  --data_path ../dataset/pretrain_smoke.jsonl \
  --save_weight pt_smoke \
  --from_weight none \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --epochs 1 \
  --batch_size 2 \
  --accumulation_steps 4 \
  --max_seq_len 256 \
  --learning_rate 5e-4 \
  --dtype float16 \
  --num_workers 2 \
  --log_interval 10 \
  --save_interval 100 \
  2>&1 | tee ../runs/PT-00/train.log



