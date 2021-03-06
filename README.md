# TSQR on TensorCores

This library provides a TSQR function which works on NVIDIA TensorCores.
The function compute a QR factorization of a m x n matrix (n <= 128).

## Supported GPUs
- Ampere (sm_80, sm_86)

## Requirements
- C++ >= 17 (This library uses `if constexpr`)
- CUDA >= 11.2
- CMake >= 3.18

## Presentation
- Hiroyuki Ootomo, Rio Yokota, "TSQR on TensorCores with error correction", SIAM CSE'21 [[slide]](https://static.momo86.net/f/1/cse21-slide)

## Supported computation mode

|  Mode name            | TensorCore | Error correction |
|:----------------------|:-----------|:-----------------|
|`fp32_fp16_hmma_cor`   | HMMA-FP16  | Yes              |
|`fp32_fp16_hmma_no_cor`| HMMA-FP16  | No               |
|`fp32_tf32_hmma_cor`   | HMMA-TF32  | Yes              |
|`fp32_tf32_hmma_no_cor`| HMMA-TF32  | No               |
|`fp32_no_tc`           | No         | No               |

## Hou to use
1. Clone this repository
```bash
git clone https://github.com/enp1s0/tsqr-tc
cd tsqr-tc
git submodule update --init --recursive
```

2. Build the library
```bash
mkdir build
cd build
cmake ..
make
```

3. Link the output library to your program

e.g.
```bash
nvcc -L/path/to/tsqr-tc/build -I/path/to/tsqr-tc/include -ltsqr-tc ...
```

## Sample code
```cuda
// Compute QR factorization of a m x n matrix
#include <tsqr_tc/tsqr.hpp>

constexpr compute_mode = mtk::tsqr_tc::compute_mode::fp32_fp16_hmma_cor;

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
