CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --main_process_port 20655 --num_processes 3 --mixed_precision "fp16" \
  trainer/tutorial_train_base_clip.py \
  --pretrained_model_name_or_path="/data1/JM/code/IP-Adapter-main/pretrain_models/stable-diffusion-v1-5" \
  --image_encoder_path="/data1/JM/code/BrushNet/pretrain_model/image_encoder" \
  --data_json_file="/data1/JM/code/IP-Adapter-main/dataset/MSRA-10K/data.json" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=8 \
  --dataloader_num_workers=8 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="exp/base_clip" \
  --save_steps=5000 \
  --enable_xformers_memory_efficient_attention \
  --num_train_epochs 100