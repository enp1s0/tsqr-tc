NVCC=nvcc

SRCDIR=src
OBJDIR=obj
LIBDIR=lib
TESTDIR=test
INCLUDEDIR=include

NVCCFLAGS=-std=c++17 -t0
NVCCFLAGS+=-I./$(SRCDIR)/cutf/include
NVCCFLAGS+=-I./$(SRCDIR)/wmma_extension/include
NVCCFLAGS+=-I./$(SRCDIR)/hmma.f32.f32/include
NVCCFLAGS+=-I./$(SRCDIR)/gemm_core_cuh/include
NVCCFLAGS+=-gencode arch=compute_86,code=sm_86
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80

NVCCFLAGS+=-I./$(INCLUDEDIR)

TARGET=libtsqr-tc.a
SRCS=batchedqr.cu tsqr_buffer.cu tsqr.cu
OBJS=$(SRCS:%.cu=$(OBJDIR)/%.o)
DLINKOBJS=$(SRCS:%.cu=$(OBJDIR)/%.dlink.oo)
HEADERS=$(shell find $(INCLUDEDIR) -name '*.cuh' -o -name '*.hpp' -o -name '*.h')

PROFILING=NO

ifeq ($(PROFILING), YES)
NVCCFLAGS+=-lineinfo --ptxas-options=-v
endif

all: $(LIBDIR)/$(TARGET)
	make tests

$(LIBDIR)/$(TARGET): $(OBJS)
	[ -d $(LIBDIR) ] || mkdir $(LIBDIR)
	$(NVCC) $+ $(NVCCFLAGS) -o $@ -lib

$(OBJDIR)/%.dlink.oo: $(OBJDIR)/%.o
	$(NVCC) $< $(NVCCFLAGS) -o $@ -dlink

$(OBJDIR)/%.o: $(SRCDIR)/%.cu  $(HEADERS)
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(NVCC) $< $(NVCCFLAGS) -o $@ -dc -c

clean:
	rm -f $(OBJDIR)/*
	rm -f $(LIBDIR)/*
	cd $(TESTDIR);make clean

tests:
	cd $(TESTDIR);make
