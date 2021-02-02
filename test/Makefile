NVCC=nvcc

INCDIR=../include
LIBDIR=../lib

NVCCFLAGS=-std=c++17 -I$(INCDIR) -L$(LIBDIR) -ltsqr-tc -I../src/cutf/include -Xcompiler="-fopenmp"

TARGET=qr256x128.test

all: $(TARGET)

%.test:%.cu $(LIBDIR)/libtsqr-tc.a
	$(NVCC) $< $(NVCCFLAGS) -o $@

clean:
	rm -f *.test
