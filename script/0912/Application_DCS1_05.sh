

# # 05

echo 05

CUDA_VISIBLE_DEVICES=6 python fine-tune.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/DCS1/05" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/17/DCS1/05" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/17/DCS1/05" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth"

