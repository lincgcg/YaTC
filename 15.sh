# CL 36

# 6.67e-4

# 50

CUDA_VISIBLE_DEVICES=4 python fine-tune_CL_finetuen.py --blr 6.67e-4 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/36" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/36" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/36/checkpoint-epoch50.pth" --project_name DCS_YaTC_application_CL --name 36_50_6.67e-4

# 100

CUDA_VISIBLE_DEVICES=4 python fine-tune_CL_finetuen.py --blr 6.67e-4 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/36" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/36" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/36/checkpoint-epoch100.pth" --project_name DCS_YaTC_application_CL --name 36_100_6.67e-4


# 150

CUDA_VISIBLE_DEVICES=4 python fine-tune_CL_finetuen.py --blr 6.67e-4 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/36" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/36" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/36/checkpoint-epoch150.pth" --project_name DCS_YaTC_application_CL --name 36_150_6.67e-4

# 200

CUDA_VISIBLE_DEVICES=4 python fine-tune_CL_finetuen.py --blr 6.67e-4 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/36" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/36" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/36/checkpoint-epoch200.pth" --project_name DCS_YaTC_application_CL --name 36_200_6.67e-4_wPH

# CL 37

# 2e-3

# 50

CUDA_VISIBLE_DEVICES=4 python fine-tune_CL_finetuen.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/37" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/37" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/37/checkpoint-epoch50.pth" --project_name DCS_YaTC_application_CL --name 37_50_2e-3_wPH

# 100

CUDA_VISIBLE_DEVICES=4 python fine-tune_CL_finetuen.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/37" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/37" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/37/checkpoint-epoch100.pth" --project_name DCS_YaTC_application_CL --name 37_100_2e-3_wPH


# 150

CUDA_VISIBLE_DEVICES=4 python fine-tune_CL_finetuen.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/37" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/37" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/37/checkpoint-epoch150.pth" --project_name DCS_YaTC_application_CL --name 37_150_2e-3_wPH


# CL-pretrain

# CL 41

CUDA_VISIBLE_DEVICES=2,3,4 python  fine-tune_CL_pretrain.py --blr 6.25e-5 --epochs 200 --batch_size 2048 --nb_classes 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/CL" --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/40" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/40" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth" --temperature 0.3 --project_name YaTC_CL_Pretrain --name 40_6.25e-5_200_0.3

# CL 42

CUDA_VISIBLE_DEVICES=2,3,4 python  fine-tune_CL_pretrain.py --blr 6.25e-5 --epochs 200 --batch_size 2048 --nb_classes 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/CL" --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/41" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/41" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth" --temperature 0.4 --project_name YaTC_CL_Pretrain --name 41_6.25e-5_200_0.4

# CL 43

CUDA_VISIBLE_DEVICES=2,3,4 python  fine-tune_CL_pretrain.py --blr 6.25e-5 --epochs 200 --batch_size 2048 --nb_classes 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/CL" --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/42" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/42" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth" --temperature 0.5 --project_name YaTC_CL_Pretrain --name 42_6.25e-5_200_0.5