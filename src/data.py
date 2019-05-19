#!/usr/bin/python3
import sys
import random
import numpy

with open(sys.argv[1], "w") as f:
    for i in range(int(sys.argv[2])):
        f.write(str(numpy.float32(random.uniform(-10.0, 10.0)))+"\t"+str(numpy.float32(random.uniform(-10.0, 10.0)))+"\t" +
                str(numpy.float32(random.uniform(-10.0, 10.0)))+"\t"+str(numpy.float32(random.uniform(-10.0, 10.0)))+"\t"+"nothing\n")
