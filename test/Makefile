NVCC=nvcc

INCDIR=../include
LIBDIR=../lib

NVCCFLAGS=-std=c++17 -I$(INCDIR) -L$(LIBDIR) -ltsqr-tc -I../src/cutf/include -Xcompiler="-fopenmp" -lcublas
NVCCFLAGS+=-gencode arch=compute_86,code=sm_86

TARGET=qr256x128.test
OBJS=utils.o

all: $(TARGET)

%.test:%.cu $(OBJS) $(LIBDIR)/libtsqr-tc.a
	$(NVCC) $< $(OBJS) $(NVCCFLAGS) -o $@

%.o:%.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

clean:
	rm -f *.test
