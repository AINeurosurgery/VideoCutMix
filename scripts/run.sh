DATASETS=('proposed_nets_videomae')
mstcn_device=0

for dataset in "${DATASETS[@]}"
do
    echo "Running for MS-TCN"
    python experiments/run_mstcn/mstcn_train_eval.py $dataset $mstcn_device
    echo "Completed run for MS-TCN"
done
python utils/compile_results.py