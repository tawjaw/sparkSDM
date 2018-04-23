from SDM import *
import numpy as np
import time

import pyspark
from pyspark import SparkConf, SparkContext

sc = pyspark.SparkContext("local", "testsdm")

min = 0
max = 15
dim = 500
tests=[(1000,94,"SDM_1000"),(10000,92,"SDM_10000"),(100000,91,"SDM_100000")]

#100 data words to write and read from memory
testData = np.random.randint(min,max,(100,dim))


"""
create the 3 modes of SDM memory with the 3 numHL settings stated above.
Write and read 100 data point randomly generated in testData.
for 'nofile' mode create will be used to open the memory

nofile  parquet-hdf5   hdf5
[0,         0,         0] --> tests[0]
[0,         0,         0] --> tests[1]
[0,         0,         0] --> tests[2]

"""
testTimeCreate = [[0,0,0]]*3
testTimeOpenWrite =[[0,0,0]]*3
testTimeOpenRead = [[0,0,0]]*3

##test creating memory
for test in range(3):
    settings = SDM.Settings(min,max,dim,tests[test][0],tests[test][1])
    file = tests[test][2]
    #no file memory
    start = time.time()
    sdm = SDM(sc,settings,mode='nofile')
    sdm.create()
    end = time.time()
    testTimeCreate[test][0] =end - start
    print("test: " + str(test)+" no file: Done")
    del sdm
    #parquet-hdf5 memory
    start = time.time()
    sdm = SDM(sc,settings,mode='parquet-hdf5')
    sdm.create(file+'-parquet-hdf5/')
    end = time.time()
    testTimeCreate[test][1] =end - start
    print("test: " + str(test)+" parquet-hdf5: Done")
    del sdm
    #hdf5 memory
    start = time.time()
    sdm = SDM(sc,settings,mode='hdf5')
    sdm.create(file+'-mmhdf5/')
    end = time.time()
    testTimeCreate[test][2] =end - start
    print("test: " + str(test)+" hdf5: Done")   
    del sdm

print(testTimeCreate)