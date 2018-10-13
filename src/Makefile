MAKEFLAGS = --no-print-directory

all:
	make clean; make gpu; make cpu

cpu:
	g++ --std=c++17 -pthread algorithm.cpp -o cpu

gpu:
	nvcc --std=c++14 -arch=sm_50 -rdc=true algorithm.cu -o gpu

.PHONY: clean

clean:
	rm -f gpu cpu
