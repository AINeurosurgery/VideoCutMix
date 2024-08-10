model_path=models/rgb_imagenet.pt
read -p "Enter in_dir: " in_dir    #### Input directory of videos
read -p "Enter out_dir: " out_dir    #### Output directory of saved features
read -p "Starting index: " start    #### Starting index of sorted list of videos
read -p "Ending index: " end    #### Ending index of sorted list of videos
read -p "Starting Epoch: " start_epochs    #### Starting epochs
read -p "Ending Epoch: " end_epochs    #### Ending epochs
read -p "I3d Type: " i3d_type    #### 0 for original or 1 for begg_end
read -p "Cuda: " cuda
read -p "Subdivision: " subdivision    #### Number of simultaneous process to run for manual batching
read -p "Window Size: " window_size    #### Size of window 

export CUDA_VISIBLE_DEVICES=${cuda}

mkdir ${out_dir}
mkdir ${out_dir}/i3d
mkdir ${out_dir}/index
mkdir ${out_dir}/json

for((i=start_epochs; i<= end_epochs; i++))
do
    mkdir ${out_dir}/i3d/${i}
    mkdir ${out_dir}/index/${i}
    mkdir ${out_dir}/json/${i}
    
    wait
    for((j=start; j < end; j+=subdivision));
    do
        if [ $(($j+$subdivision)) -gt $end ]
        then
            end_val=$(($end))
        else
            end_val=$(($j+$subdivision))
        fi
        python extract_features_without_img.py \
        --mode rgb \
        --load_model ${model_path} \
        --input_dir ${in_dir} \
        --output_dir ${out_dir} \
        --sample_mode resize \
        --window_size ${window_size} \
        --frequency 4 \
        --no-usezip \
        --start ${j} \
        --end ${end_val} \
        --seed $RANDOM \
        --epoch ${i} \
        --i3d_type ${i3d_type} &
    done
    wait
done