#!/bin/bash

#SBATCH --output=/dev/null
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20

mkdir -p logs/$TAG
git archive -o logs/$TAG/code.zip HEAD

source /etc/profile
module load anaconda/2023a-pytorch
module load cuda/11.8

export OMP_NUM_THREADS=20

python main.py \
  --arch $ARCH \
  --lr $LR \
  --width $WIDTH \
  --train_steps $TRAIN_STEPS \
  --log_dir logs/$TAG \
  1> logs/$TAG/out.log \
  2> logs/$TAG/err.log
