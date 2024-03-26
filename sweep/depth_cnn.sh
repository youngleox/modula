for DEPTH in 2 4 8 16 32 64 128 256 512 1024 2048 4096; do
for LR in 0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 4.0; do
     export TAG=rescnn_depth_sweep/$DEPTH/$LR
     export LOG_INTERVAL=100
     export SEED=0
     export BATCH_SIZE=128
     export TRAIN_STEPS=10000
     export TEST_STEPS=100
     export DATASET=cifar10
     export ARCH=resmlp
     export DEPTH=$DEPTH
     export BLOCK_DEPTH=2
     export WIDTH=32
     export OPTIM=mgd
     export LOSS=xent
     export LR=$LR
     export BETA=0.9
     export WD=0.01
     sbatch --export=ALL sweep/run.sh
done
done