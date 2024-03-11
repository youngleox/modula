for WIDTH in 32 64 128 256 512 1024 2048; do
for LR in 0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 4.0; do
     export TAG=first-go/$WIDTH/$LR
     export WIDTH=$WIDTH
     export LR=$LR
     export ARCH=resmlp
     sbatch --export=ALL sweep/run.sh
done
done
