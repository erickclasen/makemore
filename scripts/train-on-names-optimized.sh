#!/bin/bash

#0xda599a {'input_file': 'names.txt', 'work_dir': 'gridtest', 'resume': False, 'sample_only': False, 'num_workers': 4, 'max_steps': 501, 'device': 'cpu', 'seed': 3407, 'top_k': -1, 'type': 'transformer', 'n_layer': 4, 'n_head': 32, 'n_embd': 128, 'n_embd2': 64, 'batch_size': 64, 'learning_rate': 0.001, 'weight_decay': 0.001}
#0x5c5ed2 {'input_file': 'names.txt', 'work_dir': 'gridtest', 'resume': False, 'sample_only': False, 'num_workers': 4, 'max_steps': 501, 'device': 'cpu', 'seed': 3407, 'top_k': -1, 'type': 'transformer', 'n_layer': 8, 'n_head': 16, 'n_embd': 128, 'n_embd2': 32, 'batch_size': 64, 'learning_rate': 0.001, 'weight_decay': 0.001}

# Define the values for each hyperparameter
p1=(0.001) # lr
p2=(0.001) # weight decay
p3=(16) # N-heads for GPT
p4=(128) # Embed 1
p5=(32)	 # Embed 2
p6=(8) # Layers

# Baseline Test
#python3 makemore.py -i names.txt -o names-optim --max-steps 10001
python3 makemore.py -i names.txt -l $p1 -w $p2 --n-head $p3 --n-embd $p4 --n-embd2 $p5 --n-layer $p6 -b 64 --max-steps 10001
