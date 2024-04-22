if test "$3" = "adam" && test "$4" != normalize; then
    LRS=(0.00012207031 0.00024414062 0.00048828125 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125)
else
    LRS=(0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 4.0)
fi

if test "$3" = "adam"; then
    BETA2=0.99
else
    BETA2=-1
fi

WIDTHS=(32 64 128 256 512 1024 2048)
DEPTH=3

for WIDTH in ${WIDTHS[@]}; do
for LR in ${LRS[@]}; do
     export TAG=$1/$2/$3/$4/$WIDTH/$DEPTH/$LR
     export LOG_INTERVAL=100
     export SEED=0
     export BATCH_SIZE=128
     export TRAIN_STEPS=10000
     export TEST_STEPS=100
     export DATASET=$1
     export ARCH=$2
     export DEPTH=$DEPTH
     export BLOCK_DEPTH=2
     export WIDTH=$WIDTH
     export CONTEXT=256
     export NUM_HEADS=6
     export D_EMBED=$WIDTH
     export D_QUERY=$WIDTH
     export D_VALUE=$WIDTH
     export LOSS=xent
     export LR=$LR
     export BETA1=0.9
     export BETA2=$BETA2
     export WD=0.01
     if test "$4" != normalize; then
         sbatch --export=ALL sweep/run_unnormalized.sh
     else
         sbatch --export=ALL sweep/run_normalized.sh
     fi
done
done

DEPTHS=(2 4 8 16 32 64 128 256 512)
WIDTH=128

for DEPTH in ${DEPTHS[@]}; do
for LR in ${LRS[@]}; do
     export TAG=$1/$2/$3/$4/$WIDTH/$DEPTH/$LR
     export LOG_INTERVAL=100
     export SEED=0
     export BATCH_SIZE=128
     export TRAIN_STEPS=10000
     export TEST_STEPS=100
     export DATASET=$1
     export ARCH=$2
     export DEPTH=$DEPTH
     export BLOCK_DEPTH=2
     export WIDTH=$WIDTH
     export CONTEXT=256
     export NUM_HEADS=6
     export D_EMBED=$WIDTH
     export D_QUERY=$WIDTH
     export D_VALUE=$WIDTH
     export LOSS=xent
     export LR=$LR
     export BETA1=0.9
     export BETA2=$BETA2
     export WD=0.01
     if test "$4" != normalize; then
         sbatch --export=ALL sweep/run_unnormalized.sh
     else
         sbatch --export=ALL sweep/run_normalized.sh
     fi
done
done
