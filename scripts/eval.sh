# Fail the script if there is any failure.
set -e

# It requires three arguments: Model type, experiment name, and checkpoint path.
if [[ $# -eq 3 ]] ; then
    model=$1
    exp=$2
    ckpt_path=$3
else
    echo 'Model type, experiment name, or checkpoint path is missing!'
    exit 1
fi

# Create folder, copy files, and evaluate.
output=output/$exp

if [ ! -d "$output" ]; then
    echo "Folder $output does not exist. Create folder and copy files ..."
    mkdir -p $output/src
    mkdir -p $output/checkpoints
    cp -r src/* $output/src/        # Copy source files.
    cp $0 $output/run.bash          # Copy bash script.

    echo "Start evaluating $model ..." | tee -a $output/history.txt
    PYTHONPATH=$PYTHONPATH:$output/src python -m torch.distributed.launch \
        --master_port=$((1000 + RANDOM % 9999)) --nproc_per_node=1 --use_env $output/src/main.py \
        --data-path ./data/ImageNet \
        --model $model --batch-size 128 \
        --output_dir=$output \
        --resume $ckpt_path \
        --eval | tee -a $output/history.txt
else
    echo "Folder $output already exists. Please remove the folder and re-run the script."
fi