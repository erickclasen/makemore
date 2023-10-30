#!/bin/bash
#https://samanemami.medium.com/how-to-write-our-own-grid-search-in-bash-script-91ef69ce110


#usage: makemore.py [-h] [--input-file INPUT_FILE] [--work-dir WORK_DIR]
#                   [--resume] [--sample-only] [--num-workers NUM_WORKERS]
#                   [--max-steps MAX_STEPS] [--device DEVICE] [--seed SEED]
#                   [--top-k TOP_K] [--type TYPE] [--n-layer N_LAYER]
#                   [--n-head N_HEAD] [--n-embd N_EMBD] [--n-embd2 N_EMBD2]
#                   [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE]
#                   [--weight-decay WEIGHT_DECAY]


#{'input_file': 'names.txt', 'work_dir': 'gridtest', 'resume': False, 'sample_only': False, 'num_workers': 1, 'max_steps': 501, 'device': 'cpu', 'seed': 3407, 'top_k': -1, 'type': 'transformer', 'n_layer': 4, 'n_head': 32, 'n_embd': 128, 'n_embd2': 64, 'batch_size': 64, 'learning_rate': 0.001, 'weight_decay': 0.01}

# Define the range of values for each hyperparameter


param_1=(0.01 0.005 0.001 0.0005 0.0001) # lr
param_2=(0.01) # weight decay
param_3=(1 2 3 4 6 ) # N-heads for GPT
param_4=(4 8 12 16 24 32) # Embed 1
param_5=(32) # Embed 2
param_6=(1 2 3 4 6) # Layers


# Loop through each combination of hyperparameters
for p1 in "${param_1[@]}"; do
  for p2 in "${param_2[@]}"; do
    for p3 in "${param_3[@]}"; do
      for p4 in "${param_4[@]}"; do
        for p5 in "${param_5[@]}"; do
	  for p6 in "${param_6[@]}"; do
                # Train the model using the current combination of hyperparameters
                #model=$(train_model $p1 $p2)
                python3 makemore.py -i math_dataset.txt -o gridtest -l $p1 -w $p2 --n-head $p3 --n-embd $p4 --n-embd2 $p5 --n-layer $p6 --max-steps 501 -b 32 -m 2000
          done
        done
      done
    done
  done
done

