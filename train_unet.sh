CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 20696 --num_processes 1 --mixed_precision "fp16" \
  trainer/tutorial_train_base_clip_unet.py \
  --pretrained_model_name_or_path="pretrain_models/stable-diffusion-v1-5" \
  --data_json_file="dataset/MSRA-10K_inpaint/data.json" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=4 \
  --dataloader_num_workers=8 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="exp/base_unet" \
  --save_steps=5000 \
  --enable_xformers_memory_efficient_attention \
  --num_train_epochs 100
