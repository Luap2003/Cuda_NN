#!/bin/bash

# Filename: run_neural_net.sh

# Ensure the script exits if any command fails
set -e

# Loop from 1 to 100
for i in {1..10}
do
    echo "Running neural_net iteration: $i"
    ./bin/neural_net
done

echo "Completed running neural_net 100 times."
