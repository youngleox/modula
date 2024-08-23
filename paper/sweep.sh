#!/bin/bash

export BATCH_SIZE=128
export TRAIN_STEPS=10000
export TEST_STEPS=1000
export DATASET=cifar10

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

FINAL_DEPTHS=(1 2 4 8 16)
LR=(0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 4.0)

for LR in ${LRS[@]}; do
python main.py \
	--log_dir logs/$TAG \
	--batch_size $BATCH_SIZE \
	--train_steps $TRAIN_STEPS \
	--test_steps $TEST_STEPS \
	--dataset $DATASET \
	--arch $M \
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


