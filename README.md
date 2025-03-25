# Cuda_NN
Code for a runtime comparison of the implementation of a
neural network with CUDA and cuDNN against the implementation of the same
network with Tensorflow/Keras for the Geocomputing Course of the Uni Potsdam.

# Installation instructions
## Dependencies
- make
- cuda
- cuDNN

## Build
Run
```sh
$ make all
```

## Test
Run
```sh
$ make test
```

# Usage guide
Run the executable under `bin/neural_net`.

# Documentation
Make sure doxygen is installed.

To generate the documentation run
```sh
$ make docs
```

# Benchmarking
To benchmark against TF for multiple Parameters:
```
./benchmark_cuda.sh
```

To analyze the program further use:
```
nsys profile -t cuda,osrt --stats=true -o my_report ./bin/neural_ne
```

# License
This work is licensed under the MIT licSense.
# Contribution guidelines
For now there will be no contributions welcomed, because the project is to
be graded.
