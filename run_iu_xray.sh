python main.py \
--image_dir /user-data/mydata/IU-Xray/images/images_normalized/ \
--ann_path data/iu_xray/iuxray_label_40_annotation.json \
--dataset_name iu_xray \
--num_class 40 \
--max_seq_length 60 \
--threshold 3 \
--batch_size 100 \
--epochs 100 \
--save_dir results/iu_xray/mesh2/ \
--step_size 500 \
--gamma 0.1 \
--seed 9223 \
--lr_ed 5e-3 \
--img_size 384 \
--input_features 'mesh_feature' \
--test True \
--resume results/iu_xray/mesh2/model_best.pth
#--pre_resume /query2labels/output_iu/model_best.pth.tar
# --resume results/iu_xray/mesh2/current_checkpoint.pth
# --pre_resume /query2labels/output_iu/model_best.pth.tar
# --resume results/iu_xray/mesh1/model_best.pth


