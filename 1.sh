echo "28 finetune"

echo "blr = 2e-3 epoch num = 200"

CUDA_VISIBLE_DEVICES=0 python fine-tune_CL_finetuen.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/28" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/28" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/28/checkpoint-epoch200.pth" --project_name DCS_YaTC_application_CL --name 28_200_2e-3

echo "blr = 1e-4 epoch num = 200"

CUDA_VISIBLE_DEVICES=0 python fine-tune_CL_finetuen.py --blr 1e-4 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/28" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/28" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/28/checkpoint-epoch200.pth" --project_name DCS_YaTC_application_CL --name 28_200_1e-4

echo "blr = 2e-3 epoch num = 100"

CUDA_VISIBLE_DEVICES=0 python fine-tune_CL_finetuen.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/28" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/28" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/28/checkpoint-epoch100.pth" --project_name DCS_YaTC_application_CL --name 28_100_2e-3

echo "blr = 1e-4 epoch num = 100"

CUDA_VISIBLE_DEVICES=0 python fine-tune_CL_finetuen.py --blr 1e-4 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/28" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/28" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/28/checkpoint-epoch100.pth" --project_name DCS_YaTC_application_CL --name 28_100_1e-4
