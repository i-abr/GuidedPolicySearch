#!/bin/zsh

for seed in 0 1 42 666 1234
do
    python3 main.py \
                --env "HopperBulletEnv" \
                --max_steps 2000 \
                --max_frames 20000 \
                --horizon 10 \
                --frame_skip 4 \
                --seed $seed \
                --render
    echo "trial $i out of 2"
done
