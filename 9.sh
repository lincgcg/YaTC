# CL 34

CUDA_VISIBLE_DEVICES=2,3,4 python  fine-tune_CL_pretrain.py --blr 6.25e-5 --epochs 200 --batch_size 2048 --nb_classes 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/CL" --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/34" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/34" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth" --temperature 0.01 --project_name YaTC_CL_Pretrain --name 34_6.25e-5_200_0.01

# CL 35

CUDA_VISIBLE_DEVICES=2,3,4 python  fine-tune_CL_pretrain.py --blr 6.25e-5 --epochs 200 --batch_size 2048 --nb_classes 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/CL" --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/35" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/35" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth" --temperature 0.2 --project_name YaTC_CL_Pretrain --name 35_6.25e-5_200_0.2

# CL 36

CUDA_VISIBLE_DEVICES=2,3,4 python  fine-tune_CL_pretrain.py --blr 6.25e-5 --epochs 200 --batch_size 2048 --nb_classes 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/CL" --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/36" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/36" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth" --temperature 0.5 --project_name YaTC_CL_Pretrain --name 36_6.25e-5_200_0.5