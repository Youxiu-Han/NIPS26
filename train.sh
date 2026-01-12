#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

# 循环执行 train.py 脚本
for i in 0; do
  config_file="config/textmatch_cub200_2000_0.yaml"
  # 检查配置文件是否存在
  if [ -f "$config_file" ]; then
    echo "Running python train.py with config: ${config_file}"
    python train.py --c "$config_file" 
  else
    echo "Config file not found: ${config_file}"
  fi
done

