#!/bin/bash

# Loop 100 times
for ((i = 1; i <= 1000; i++)); do
    random_number=$RANDOM
    #echo "Random number $i: $random_number"
    #python3 makemore.py -e 1 --seed $random_number
    python3 makemore.py --seed $random_number --max-steps 501

done
