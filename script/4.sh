CUDA_VISIBLE_DEVICES=3 python fine-tune_CL_finetuen.py --blr 1e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/46" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/46" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/46/checkpoint-epoch200.pth" --project_name DCS_YaTC_application_CL --name 46_200_1e-3_1e-7 --min_lr 1e-7

CUDA_VISIBLE_DEVICES=3 python fine-tune_CL_finetuen.py --blr 1e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/47" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/47" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/47/checkpoint-epoch200.pth" --project_name DCS_YaTC_application_CL --name 47_200_1e-3_1e-7 --min_lr 1e-7