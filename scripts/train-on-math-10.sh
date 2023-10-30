#!/bin/bash


# Define the values for each hyperparameter
p1=(0.01) # lr
p2=(0.01) # weight decay
p3=(4) # N-heads for GPT
p4=(8) # Embed 1
p5=(32)  # Embed 2
p6=(1) # Layers


# Baseline Test
python3 makemore.py -i math_dataset.txt -o math-10 -l $p1 -w $p2 --n-head $p3 --n-embd $p4 --n-embd2 $p5 --n-layer $p6 -b 32 --max-steps 100001 -e 10
