# TSQR on TesnorCores

This library provides a TSQR function which works on NVIDIA TensorCores.
The function compute a QR factorization of a m x n matrix (n <= 128).

## Supported GPUs
- Volta
- Ampere

This library does not support Turing architecture because it does not have enough shared memory.

## Requirements
- C++ >= 17 (This library uses `if constexpr`)
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

## Sample code
```cuda
// Compute QR factorization of a m x n matrix
#include <tsqr_tc/tsqr.hpp>

constexpr compute_mode = mtk::tsqr_tc::compute_mode::fp32_hmma_cor;

int main() {
	// ...
	mtk::tsqr_tc::tsqr_buffer<compute_mode> tsqr_buffer(m, n);
	tsqr_buffer.allocate();
	tsqr_buffer.set_indices();

	mtk::tsqr_tc::tsqr<compute_mode>(
			dQ, m,
			dR, n,
			dA, m,
			m, n,
			tsqr_buffer
			);
}

```

## Algorithm
This library computes TSQR using a batch QR function.
It uses Householder QR and WY Representation to compute each QR factorization in batch QR.


## License
MIT
