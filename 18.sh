# CL 39

# 2e-3

# 100

CUDA_VISIBLE_DEVICES=7 python fine-tune_CL_finetuen.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/39" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/39" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/39/checkpoint-epoch100.pth" --project_name DCS_YaTC_application_CL --name 39_100_2e-3_wPH


# 150

CUDA_VISIBLE_DEVICES=7 python fine-tune_CL_finetuen.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/39" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/39" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/39/checkpoint-epoch150.pth" --project_name DCS_YaTC_application_CL --name 39_150_2e-3_wPH

# 200

CUDA_VISIBLE_DEVICES=7 python fine-tune_CL_finetuen.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/39" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/39" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/39/checkpoint-epoch200.pth" --project_name DCS_YaTC_application_CL --name 39_200_2e-3_wPH

# CL 39

# 6.67e-4

# 50

CUDA_VISIBLE_DEVICES=7 python fine-tune_CL_finetuen.py --blr 6.67e-4 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/39" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/39" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/39/checkpoint-epoch50.pth" --project_name DCS_YaTC_application_CL --name 39_50_6.67e-4_wPH

# 100

CUDA_VISIBLE_DEVICES=7 python fine-tune_CL_finetuen.py --blr 6.67e-4 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/39" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/39" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/39/checkpoint-epoch100.pth" --project_name DCS_YaTC_application_CL --name 39_100_6.67e-4_wPH


# 150

CUDA_VISIBLE_DEVICES=7 python fine-tune_CL_finetuen.py --blr 6.67e-4 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/39" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/39" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/39/checkpoint-epoch150.pth" --project_name DCS_YaTC_application_CL --name 39_150_6.67e-4_wPH

# 200

CUDA_VISIBLE_DEVICES=7 python fine-tune_CL_finetuen.py --blr 6.67e-4 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/39" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/39" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/39/checkpoint-epoch200.pth" --project_name DCS_YaTC_application_CL --name 39_200_6.67e-4_wPH

# CL-pretrain

# CL 43

CUDA_VISIBLE_DEVICES=5,6,7 python  fine-tune_CL_pretrain.py --blr 6.25e-5 --epochs 200 --batch_size 2048 --nb_classes 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/CL" --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/43" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/43" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth" --temperature 0.6 --project_name YaTC_CL_Pretrain --name 43_6.25e-5_200_0.6

# CL 44

CUDA_VISIBLE_DEVICES=5,6,7 python  fine-tune_CL_pretrain.py --blr 6.25e-5 --epochs 200 --batch_size 2048 --nb_classes 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/CL" --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/44" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/44" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth" --temperature 0.7 --project_name YaTC_CL_Pretrain --name 44_6.25e-5_200_0.7

# CL 45

CUDA_VISIBLE_DEVICES=5,6,7 python  fine-tune_CL_pretrain.py --blr 6.25e-5 --epochs 200 --batch_size 2048 --nb_classes 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/CL" --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/45" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/45" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth" --temperature 0.8 --project_name YaTC_CL_Pretrain --name 45_6.25e-5_200_0.8