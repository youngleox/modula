for DEPTH in 1 2 4 8 16 32 64 128 256 512; do
for LR in 0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 4.0; do
     export TAG=sweep-depth/$DEPTH/$LR
     export WIDTH=512
     export DEPTH=$DEPTH
     export LR=$LR
     export ARCH=resmlp
     export TRAIN_STEPS=10000
     sbatch --export=ALL sweep/run.sh
done
done
