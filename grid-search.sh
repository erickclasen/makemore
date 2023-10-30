'''
https://samanemami.medium.com/how-to-write-our-own-grid-search-in-bash-script-91ef69ce110
'''


# Define the range of values for each hyperparameter
param_1=(1e-2 1e-3 1e-4 1e-5 1e-6) # lr
param_2=(0.1 0.001 0.0001) # weight decay

# Loop through each combination of hyperparameters
for p1 in "${param_1[@]}"; do
  for p2 in "${param_2[@]}"; do
    # Train the model using the current combination of hyperparameters
    #model=$(train_model $p1 $p2)
    python3 makemore.py -i unixdict.txt -o gridtest -l $p1 -w $p2 --max-steps 501 -n 1 -b 64

  done
done

