# CL 37

CUDA_VISIBLE_DEVICES=5,6,7 python  fine-tune_CL_pretrain.py --blr 6.25e-5 --epochs 200 --batch_size 2048 --nb_classes 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/CL" --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/37" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/37" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth" --temperature 1.0 --project_name YaTC_CL_Pretrain --name 37_6.25e-5_200_1.0

# CL 38

CUDA_VISIBLE_DEVICES=5,6,7 python  fine-tune_CL_pretrain.py --blr 6.25e-5 --epochs 200 --batch_size 2048 --nb_classes 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/CL" --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/38" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/38" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth" --temperature 5.0 --project_name YaTC_CL_Pretrain --name 38_6.25e-5_200_5.0

# CL 39

CUDA_VISIBLE_DEVICES=5,6,7 python  fine-tune_CL_pretrain.py --blr 6.25e-5 --epochs 200 --batch_size 2048 --nb_classes 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/CL" --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/39" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/39" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth" --temperature 10.0 --project_name YaTC_CL_Pretrain --name 39_6.25e-5_200_10.0