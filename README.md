# TSQR on TesnorCores

This library provides a TSQR function which works on NVIDIA TensorCores.

## Requirements
- C++ >= 17
- CUDA >= 9.2

## Hou to use
1. Clone this repository
```bash
git clone https://github.com/enp1s0/tsqr-tc
```

2. Build the library
```bash
make
```

3. Link to your program
```bash
nvcc -L/path/to/tsqr-tc/lib -I/path/to/tsqr-tc/include -ltsqr-tc ...
```

## Algorithm
This library computes TSQR using a batch QR function.
We use Householder QR to compute each QR factorization in batch QR.
To compute QR factorization efficiently we use WY representation.


## License
MIT
