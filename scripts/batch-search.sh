'''
https://samanemami.medium.com/how-to-write-our-own-grid-search-in-bash-script-91ef69ce110
'''
echo "Batch Size optimization, batch size, timing date to date." > timing.txt

# Define the range of values for each hyperparameter
param_1=(16 32 64 128) # batch size
#param_2=(0.1 0.001 0.0001) # weight decay

# Loop through each combination of hyperparameters
for p1 in "${param_1[@]}"; do
    # Train the model using the current combination of hyperparameters
    #model=$(train_model $p1 $p2)
    echo $p1 >> timing.txt
    date >> timing.txt
    #python3 makemore.py -i names.txt -o gridtest -b $p1 -e 1.98
    python3 makemore.py -i math_dataset.txt -o gridtest -b $p1 -t 0.54 #1.3
    date >> timing.txt
    echo "----------------------" >> timing.txt 
done

