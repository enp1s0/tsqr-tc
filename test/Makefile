NVCC=nvcc

INCDIR=../include
LIBDIR=../lib

NVCCFLAGS=-std=c++17 -I$(INCDIR) -L$(LIBDIR) -ltsqr-tc -I../src/cutf/include -Xcompiler="-fopenmp" -lcublas -lcusolver
NVCCFLAGS+=-gencode arch=compute_86,code=sm_86
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80

TARGET=qr256x256.test tsqr.test batchqr.test
OBJS=utils.o

all: $(TARGET)

%.test:%.cu $(OBJS) $(LIBDIR)/libtsqr-tc.a
	$(NVCC) $< $(OBJS) $(NVCCFLAGS) -o $@

%.o:%.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

clean:
	rm -f *.test
