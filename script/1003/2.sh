# CL 35

# 150

CUDA_VISIBLE_DEVICES=1 python fine-tune_CL_finetuen.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/35" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/35" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/35/checkpoint-epoch150.pth" --project_name DCS_YaTC_application_CL --name 35_150_2e-3

# 200

CUDA_VISIBLE_DEVICES=1 python fine-tune_CL_finetuen.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/35" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/35" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/35/checkpoint-epoch200.pth" --project_name DCS_YaTC_application_CL --name 35_200_2e-3