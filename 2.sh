echo "27 pretrain"

CUDA_VISIBLE_DEVICES=5,6,7 python  fine-tune_CL_pretrain.py --blr 5e-5 --epochs 200 --batch_size 2048 --nb_classes 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/CL" --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/27" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/27" --finetune "/data0/mxy/linchungang/YaTC_file/pretrain_file/YaTC_pretrained_model.pth" --project_name YaTC_CL_Pretrain --name 5e-5_200


echo "27 finetune"

echo "blr = 2e-3 epoch num = 200"

CUDA_VISIBLE_DEVICES=7 python fine-tune_CL_finetuen.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/27" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/27" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/27/checkpoint-epoch200.pth" --project_name DCS_YaTC_application_CL --name 27_200_2e-3

echo "blr = 1e-4 epoch num = 200"

CUDA_VISIBLE_DEVICES=7 python fine-tune_CL_finetuen.py --blr 1e-4 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/27" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/27" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/27/checkpoint-epoch200.pth" --project_name DCS_YaTC_application_CL --name 27_200_1e-4

echo "blr = 2e-3 epoch num = 100"

CUDA_VISIBLE_DEVICES=7 python fine-tune_CL_finetuen.py --blr 2e-3 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/27" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/27" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/27/checkpoint-epoch100.pth" --project_name DCS_YaTC_application_CL --name 27_100_2e-3

echo "blr = 1e-4 epoch num = 100"

CUDA_VISIBLE_DEVICES=7 python fine-tune_CL_finetuen.py --blr 1e-4 --epochs 200 --batch_size 192 --data_path "/data0/mxy/linchungang/YaTC_file/YaTC_datasets/ISXWVPN/17/woDCS/01" --nb_classes 17 --output_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/27" --log_dir "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/epoch_10/17/CL/demo/27" --finetune "/data0/mxy/linchungang/YaTC_file/output/ISXWVPN/CL/17/demo/27/checkpoint-epoch100.pth" --project_name DCS_YaTC_application_CL --name 27_100_1e-4