ckpt_path=/home/mrinal_temp/nets/CVPR-video/extraction/VideoMAEv2/ckpt/vit_g_hybrid_pt_1200e_k710_ft.pth

read -p "Enter starting epoch: " start_epoch
read -p "Enter ending epoch: " end_epoch
read -p "Starting index: " start_ind
read -p "Ending index: " end_ind
read -p "Enter in_dir: " in_dir
read -p "Enter out_dir: " out_dir
read -p "Enter cuda device: " cuda
read -p "Enter extraction type '0 for original and 1 for begg_end': " feat_type

mkdir $out_dir
mkdir $out_dir/i3d
mkdir $out_dir/index

for((i = $start_epoch; i <= $end_epoch; i++))
do
    mkdir $out_dir/i3d/$i
    mkdir $out_dir/index/$i
    wait
    python extract_tad_feature.py --data_path=$in_dir \
    --save_path=$out_dir \
    --ckpt_path=$ckpt_path \
    --type=${feat_type} \
    --cuda=$cuda \
    --epoch=${i} \
    --start=${start_ind} \
    --end=${end_ind}
    wait
done