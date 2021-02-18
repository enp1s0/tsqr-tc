# TSQR on TesnorCores

This library provides a TSQR function which works on NVIDIA TensorCores.

## Supported GPUs
- Volta
- Ampere

This library does not support Turing architecture because it does not have enough shared memory.

## Requirements
- C++ >= 17
- CUDA >= 11.2

## Supported computation mode

|  Mode name    | TensorCore | Error correction |
|:--------------|:-----------|:-----------------|
|`fp32_hmma_cor`| HMMA       | Yes              |

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
