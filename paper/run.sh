#!/bin/bash

export M=resmlp2
export L=naw
export W=64

export D=4

export I=1
export IM=1.0

export B=1
export BM=1.0

export F=1
export FM=1.0

export TAG=$M_$L/$W_$D_$I_$IM_$B_$BM_$F_$FM/$LR

FINAL_DEPTHS = (1 2 4 8 16)
LR = (0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 4.0)

for DEPTH in ${FINAL_DEPTHS[@]}; do
for LR in ${LRS[@]}; do

python paper/main.py \
	--log_dir logs/$TAG \
	--layer $L \
	--width $W \
	--depth $D \	
	--initial_depth $I \
	--block_depth $B \
	--final_depth $F \	
	--initial_mass $IM \
	--body_mass $BM \	
	--final_mass $FM \
	--lr $LR 

done
done

python paper/main.py --depth $D --block_depth $B --initial_depth $I --initial_mass $IM --final_depth $F --final_mass $FM  --lr 0.0625

python paper/main.py --depth $D --block_depth $B --initial_depth $I --initial_mass $IM --final_depth $F --final_mass $FM  --lr 0.125

python paper/main.py --depth $D --block_depth $B --initial_depth $I --initial_mass $IM --final_depth $F --final_mass $FM  --lr 0.25

python paper/main.py --depth $D --block_depth $B --initial_depth $I --initial_mass $IM --final_depth $F --final_mass $FM  --lr 0.5

python paper/main.py --depth $D --block_depth $B --initial_depth $I --initial_mass $IM --final_depth $F --final_mass $FM  --lr 1.0

python paper/main.py --depth $D --block_depth $B --initial_depth $I --initial_mass $IM --final_depth $F --final_mass $FM  --lr 2.0

python paper/main.py --depth $D --block_depth $B --initial_depth $I --initial_mass $IM --final_depth $F --final_mass $FM  --lr 4.0

export TAG=$1/$2/$3/$4/$WIDTH/$DEPTH/$LR
export LOG_INTERVAL=100
export SEED=0
export BATCH_SIZE=128
export TRAIN_STEPS=10000
export TEST_STEPS=1000
export DATASET=cifar10
export ARCH=resmlp2

export WIDTH=64
export CONTEXT=128
export NUM_HEADS=8
export LOSS=xent
export LR=$LR
export BETA1=0.9
export BETA2=$BETA2
export WD=0.0
export NORMALIZE=1

export DEPTH=$DEPTH
export BLOCK_DEPTH=$BLOCK_DEPTH

mkdir -p logs/$TAG

FINAL_DEPTHS = (1 2 4 8 16)
LR = (0.125 0.25 0.5 1.0 2.0 4.0)
for DEPTH in ${FINAL_DEPTHS[@]}; do
for LR in ${LRS[@]}; do

python main.py \
  --log_dir logs/$TAG \
  --log_interval $LOG_INTERVAL \
  --seed $SEED \
  --batch_size $BATCH_SIZE \
  --train_steps $TRAIN_STEPS \
  --test_steps $TEST_STEPS \
  --dataset $DATASET \
  --arch $ARCH \
  --depth $DEPTH \
  --block_depth $BLOCK_DEPTH \
  --width $WIDTH \
  --context $CONTEXT \
  --num_heads $NUM_HEADS \
  --normalize $NORMALIZE \
  --loss $LOSS \
  --lr $LR \
  --beta1 $BETA1 \
  --beta2 $BETA2 \
  --wd $WD \
  1> logs/$TAG/out.log \
  2> logs/$TAG/err.log
  
done
done
