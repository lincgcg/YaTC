
# 01

echo 01

CUDA_VISIBLE_DEVICES=6 python fine-tune.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/DCS1/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/17/DCS1/01" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/17/DCS1/01" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth"


# 02

echo 02

CUDA_VISIBLE_DEVICES=6 python fine-tune.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/DCS1/02" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/17/DCS1/02" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/17/DCS1/02" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth"


# 03

echo 03

CUDA_VISIBLE_DEVICES=6 python fine-tune.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/DCS1/03" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/17/DCS1/03" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/17/DCS1/03" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth"


# 04

echo 04

CUDA_VISIBLE_DEVICES=6 python fine-tune.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/DCS1/04" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/17/DCS1/04" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/17/DCS1/04" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth"


# # 05

# echo 05

# CUDA_VISIBLE_DEVICES=6 python fine-tune.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/DCS1/05" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/17/DCS1/05" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/17/DCS1/05" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth"

