#!/bin/bash

export BATCH_SIZE=128
export TRAIN_STEPS=10000
export TEST_STEPS=1000

export DATASET=cifar10

export M=resmlp2
export L=naw
export W=64

export D=32

export I=1
export IM=1.0

export B=2
export BM=1.0

export F=1
export FM=1.0

export BETA1=0.9
export BETA2=0.99
export WD=0.0
export SCH=cosine

FINAL_DEPTHS=(1 2 4 8 16)
LRS=(0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 4.0)
#LRS=(0.03125)
#for LR in ${LRS[@]}; do

for LR in ${LRS[0]}; do

export TAG="$BETA1"_"$BETA2"_"$WD"_"$SCH"/"$TRAIN_STEPS"_"$BATCH_SIZE"/"$M"_"$L"/"$W"_"$D"_"$I"_"$IM"_"$B"_"$BM"_"$F"_"$FM"/"$LR"
mkdir -p logs/$TAG

python main.py \
	--log_dir logs/$TAG \
	--batch_size $BATCH_SIZE \
	--train_steps $TRAIN_STEPS \
	--test_steps $TEST_STEPS \
	--log_interval $TEST_STEPS \
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
	--wd $WD \
	--beta1 $BETA1 \
	--beta2 $BETA2 \
	--lr $LR
done


