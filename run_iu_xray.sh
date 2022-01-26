python main.py \
--image_dir data/iu_xray/IU-Xray/images/images_normalized/ \
--ann_path data/iu_xray/iuxray_label_40_annotation.json \
--dataset_name iu_xray \
--num_class 40 \
--max_seq_length 60 \
--threshold 3 \
--batch_size 16 \
--epochs 100 \
--save_dir results/iu_xray/mesh/ \
--step_size 500 \
--gamma 0.1 \
--seed 9223 \
--lr_ed 5e-3 \
--test True \
--img_size 384 \
--input_features 'mesh_feature'

