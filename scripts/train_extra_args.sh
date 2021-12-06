# Fail the script if there is any failure.
set -e

# It requires at least two arguments: Model type and experiment name.
if [[ $# -ge 2 ]] ; then
    model=$1
    exp=$2
    extra_args=${@:3}
else
    echo 'Model type or experiment name is missing!'
    exit 1
fi

# Create folder, copy files, and train.
output=output/$exp

if [ ! -d "$output" ]; then
    echo "Folder $output does not exist. Create folder and copy files ..."
    mkdir -p $output/src
    mkdir -p $output/checkpoints
    cp -r src/* $output/src/        # Copy source files.
    cp $0 $output/run.bash          # Copy bash script.

    echo "Start training $model (extra args: $extra_args)..." | tee -a $output/history.txt
    PYTHONPATH=$PYTHONPATH:$output/src python -m torch.distributed.launch \
        --master_port=$((1000 + RANDOM % 9999)) --nproc_per_node=8 --use_env $output/src/main.py \
        --data-path ./data/ImageNet \
        --model $model \
        --output_dir=$output $extra_args | tee -a $output/history.txt  # Note: --batch-size argument is now required in $extra_args.
else
    echo "Folder $output already exists. Please remove the folder and re-run the script."
fi