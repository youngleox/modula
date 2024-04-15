for DEPTH in 2 4 8 16 32 64 128 256 512; do
for LR in 0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 4.0; do
     export TAG=normalized/sgd/$DEPTH/$LR
     export LOG_INTERVAL=100
     export SEED=0
     export BATCH_SIZE=128
     export TRAIN_STEPS=10000
     export TEST_STEPS=100
     export DATASET=cifar10
     export ARCH=resmlp
     export DEPTH=$DEPTH
     export BLOCK_DEPTH=2
     export WIDTH=128
     export CONTEXT=0
     export NUM_HEADS=0
     export D_EMBED=0
     export D_QUERY=0
     export D_VALUE=0
     export LOSS=xent
     export LR=$LR
     export BETA1=0.9
     export BETA2=-1
     export WD=0.01
     sbatch --export=ALL sweep/run_normalized.sh
done
done
