NVCC=nvcc

SRCDIR=src
OBJDIR=obj
LIBDIR=lib
INCLUDEDIR=include

NVCCFLAGS=-std=c++14
NVCCFLAGS+=-I./$(SRCDIR)/cutf/include
NVCCFLAGS+=-I./$(SRCDIR)/wmma_extension/include
NVCCFLAGS+=-gencode arch=compute_86,code=sm_86

NVCCFLAGS+=-I./$(INCLUDEDIR)

TARGET=libtsqr-tc.a
SRCS=batchedqr.cu
OBJS=$(SRCS:%.cu=$(OBJDIR)/%.o)
DLINKOBJS=$(SRCS:%.cu=$(OBJDIR)/%.dlink.oo)
HEADERS=$(shell find $(INCLUDEDIR) -name '*.cuh' -o -name '*.hpp' -o -name '*.h')

$(TARGET): $(OBJS)
	echo $(OBJS)
	$(NVCC) $+ $(NVCCFLAGS) -o $@

$(OBJDIR)/%.dlink.oo: $(OBJDIR)/%.o
	$(NVCC) $< $(NVCCFLAGS) -o $@ -dlink

$(OBJDIR)/%.o: $(SRCDIR)/%.cu  $(HEADERS)
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(NVCC) $< $(NVCCFLAGS) -o $@ -dc -c

clean:
	rm -f $(OBJDIR)/*
	rm -f $(LIBDIR)/*
