for DEPTH in 2 4 8 16 32 64 128; do
for LR in 0.00012207031 0.00024414062 0.00048828125 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125; do
     export TAG=resmlp_depth_sweep_adam/$DEPTH/$LR
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
     export OPTIM=adamw
     export LOSS=xent
     export LR=$LR
     export BETA=0.9
     export WD=0.01
     sbatch --export=ALL sweep/run.sh
done
done
