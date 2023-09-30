# 50

CUDA_VISIBLE_DEVICES=0 python fine-tune_CL_finetuen.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/30" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/30" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/30/checkpoint-epoch50.pth" --project_name DCS_YaTC_application_CL --name 30_50_2e-3

# 100

CUDA_VISIBLE_DEVICES=0 python fine-tune_CL_finetuen.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/30" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/30" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/30/checkpoint-epoch100.pth" --project_name DCS_YaTC_application_CL --name 30_100_2e-3

# 150

CUDA_VISIBLE_DEVICES=0 python fine-tune_CL_finetuen.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/30" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/30" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/30/checkpoint-epoch150.pth" --project_name DCS_YaTC_application_CL --name 30_150_2e-3

# 200

CUDA_VISIBLE_DEVICES=0 python fine-tune_CL_finetuen.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/30" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/30" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/30/checkpoint-epoch200.pth" --project_name DCS_YaTC_application_CL --name 30_200_2e-3


