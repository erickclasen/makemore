#!/bin/bash


# Define the values for each hyperparameter
p1=(0.01) # lr
p2=(0.01) # weight decay
p3=(3) # N-heads for GPT
p4=(12) # Embed 1
p5=(32)  # Embed 2
p6=(2) # Layers


# Baseline Test
python3 makemore.py -i math_dataset_28_sorted.txt -o math-28 -l $p1 -w $p2 --n-head $p3 --n-embd $p4 --n-embd2 $p5 --n-layer $p6 -b 32 --max-steps 100001 -e 10
