#!/bin/bash

export BATCH_SIZE=128
export TRAIN_STEPS=100000
export TEST_STEPS=10000

export DATASET=cifar10

export M=vit
export L=naw

export C=256
export P=4

export H=4
export W=64

export D=8

export I=1
export IM=1.0

export B=1
export BM=1.0

export F=1
export FM=1.0

export BETA1=0.9
export BETA2=0.99

export WD=0.0
export SCH=cosine

BMS=(0.125 0.25 0.5 1.0 2.0 4.0 8.0 16.0)

BMS0=(0.125 0.25 0.5 1.0)
BMS1=(2.0 4.0 8.0 16.0)

BMS10=(0.25 1.0 4.0 16.0)
BMS11=(0.25 1.0)
BMS12=(4.0 16.0)

BMS99=(4.0)

LRS=(0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 )

LRS0=(0.03125 0.125 0.5)

LRS1=(0.0625 0.25 1.0 )

DS=(4 8 16 32 )

DS0=(32 )

for D in ${DS0[*]}; do
for BM in ${BMS99[*]}; do
for LR in ${LRS1[*]}; do

if [ $M=="vit" ]; then
export TAG="$BETA1"_"$BETA2"_"$WD"_"$SCH"/"$TRAIN_STEPS"_"$BATCH_SIZE"/"$M"/"$P"_"$H"_"$W"_"$D"_"$IM"_"$BM"_"$FM"/"$LR"

elif [ $M=="mlp" ]; then
export TAG="$BETA1"_"$BETA2"_"$WD"_"$SCH"/"$TRAIN_STEPS"_"$BATCH_SIZE"/"$M"/"$C"_"$H"_"$W"_"$D"_"$IM"_"$BM"_"$FM"/"$LR"

else
export TAG="$BETA1"_"$BETA2"_"$WD"_"$SCH"/"$TRAIN_STEPS"_"$BATCH_SIZE"/"$M"_"$L"/"$W"_"$D"_"$I"_"$IM"_"$B"_"$BM"_"$F"_"$FM"/"$LR"

fi

mkdir -p logs/$TAG

python main.py \
	--log_dir logs/$TAG \
	--batch_size $BATCH_SIZE \
	--train_steps $TRAIN_STEPS \
	--test_steps $TEST_STEPS \
	--log_interval $TEST_STEPS \
	--dataset $DATASET \
	--arch $M \
	--context $C \
	--patch_size $P \
	--layer $L \
	--width $W \
	--depth $D \
	--num_heads $H\
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
done
done

