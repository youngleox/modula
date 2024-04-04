for DIM in 8 16 32 64 128 256 512 1024; do
for LR in 0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 4.0; do
     export TAG=gpt_d_embed_sweep_all_larger_fixed_dim_simplified_sens_1/$DIM/$LR
     export LOG_INTERVAL=100
     export SEED=0
     export BATCH_SIZE=64
     export TRAIN_STEPS=5000
     export TEST_STEPS=100
     export DATASET=shakespeare
     export ARCH=gpt
     export DEPTH=6
     export BLOCK_DEPTH=2
     export WIDTH=0
     export CONTEXT=256
     export NUM_HEADS=6
     export D_EMBED=$DIM
     export D_QUERY=$DIM
     export D_VALUE=$DIM
     export OPTIM=mgd
     export LOSS=xent
     export LR=$LR
     export BETA=0.9
     export WD=0.01
     sbatch --export=ALL sweep/run.sh
done
done
