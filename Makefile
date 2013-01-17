CXX=g++
CFLAGS=-Wall -O3 -fPIC
CFLAGS+=$(shell pkg-config --cflags opencv)
LDFLAGS=$(shell pkg-config --libs opencv)

all: rsr

rsr: 
	g++ -fPIC -o rsr src/gtsrb_experiments.cpp -I./vlfeat-0.9.16 -O3 -lshogun -lvl -L./vlfeat-0.9.16/bin/glnxa64 $(LDFLAGS)

clean:
	rm $(OBJECTS)

.PHONY: all clean
