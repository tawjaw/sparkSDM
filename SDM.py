#Author: Tawfiq Jawhar
import pyspark
import pyspark.mllib.random as random
from pyspark.mllib.linalg import *
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import pyspark.sql as sql
import numpy as np
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf, lit, col

import h5py

class SDM(object):
    sc = None
    class Settings:
        def __init__(self,min,max,dim, numHL, radius):
            """returns an instance of Settings used to build SDM.
            Paramaters:
                min: minimum range value of Integer SDM
                max: maximum range value of Integer SDM
                dim: number of dimensions of the memory
                numHL: number of hard locations allocated in the memory
            """
            for parm in [min,max,dim,numHL]:
                if not isinstance(parm, int):
                    raise TypeError("min,max,dim and numHL must be integers.")
            if not (isinstance(radius,float) or isinstance(radius,int)):
                raise TypeError("radius must be int or float.")
            if min >= max: raise AssertionError("minimum range can not be greater than the maximum range.")
            self.min = min
            self.max = max
            self.dim = dim
            self.numHL = numHL
            self.radius = radius
    
    def __init__(self, sc,min,max,dim,numHL,radius,mode='nofile', seed = None):
        """returns an instance of SDM.
            Paramaters:
                min: minimum range value of Integer SDM
                max: maximum range value of Integer SDM
                dim: number of dimensions of the memory
                numHL: number of hard locations allocated in the memory
                sc: SparkContext instance
                mode:
                    'nofile' (default) stores SDM's counters in numpy arrays
                    'parquet-hdf5' stores SDM's hard location addresses in parquet file and counters in hdf5 file
                    'hdf5' stores counters in hdf5 file and stores the random seed to generate the addresses RDD without storing the addresses in a file
        """
        SDM.sc = sc
        self.settings = SDM.Settings(min,max,dim,numHL,radius)
        if mode not in ['nofile','parquet-hdf5','hdf5']:
            raise AssertionError("mode "+mode+" not defined")
        self.mode = mode
        self.addresses = None
        self.seed = seed

        
    def __init__(self,sc,settings,mode='nofile',seed = None):
        """returns an instance of SDM.
            Paramaters:
                settings: an instance of SDM.Settings
                sc: SparkContext instance
                mode:
                    'nofile' (default) stores SDM's counters in numpy arrays
                    'parquet-hdf5' stores SDM's hard location addresses in parquet file and counters in hdf5 file
                    'hdf5' stores counters in hdf5 file and stores the random seed to generate the addresses RDD without storing the addresses in a file
        """
        SDM.sc = sc
        if not isinstance(settings,SDM.Settings):
            raise TypeError("settings is not instance of Settings.")
        self.settings = settings
        
        if mode not in ['nofile','parquet-hdf5','hdf5']:
            raise AssertionError("mode '"+mode+"' not defined")
        self.mode = mode
        self.addresses = None
        self.seed = seed
        
    @staticmethod
    def create_addresses_DF(settings, random_seed = None):
        spark = SparkSession.builder.appName("sdm").config(conf=SparkConf()).getOrCreate()
        HardLocations = random.RandomRDDs.uniformVectorRDD(SDM.sc, settings.numHL, settings.dim, seed=random_seed)\
            .map(lambda x: (x*(settings.max-settings.min) + settings.min).astype(int))\
            .zipWithIndex()\
            .map(lambda row: (row[1], DenseVector(row[0])))\
            .toDF(["id", "address"])
        return HardLocations
    

    
    @staticmethod
    def activated_hard_locations(df, word, radius, max):
        """returns a list of the index of activated hard locations using modular euclidean distance."""
        def euclidean_distance(arr1,arr2):
            """Returns modular euclidean distance"""
            sum = 0
            for i in range(len(arr1)):
                temp1 = np.abs(arr1[i]-arr2[i])
                temp2 = max - temp1
                tempDist = np.square(temp1) if temp1 < temp2 else np.square(temp2)
                sum = sum + tempDist
            return np.sqrt(sum)
        
        spark = SparkSession.builder.appName("sdm").config(conf=SparkConf()).getOrCreate()
        distance_udf = udf(lambda x: float(euclidean_distance(x, word)), FloatType())
        df = df.withColumn('distances', distance_udf(col('address')))
        return df.filter(df.distances < lit(radius)).select("id").rdd.map(lambda x: x[0]).collect()
    
    
    @staticmethod
    def write_addresses(df, file, mode):
        spark = SparkSession.builder.appName("sdm").config(conf=SparkConf()).getOrCreate()
        df.write.parquet(file, mode= mode)

    def activated_HL(self,word, radius = None):
        if radius == None: radius = self.settings.radius
        activated = SDM.activated_hard_locations(self.addresses, word, radius, self.settings.max)
        return activated

    def create(self,file=None, overwrite = False):
        """Creates an empty SDM memory.
        Parameters:
            file: (default = None) if the mode of SDM is 'parquet-hdf5' or 'hdf5' the file is the directory where the memory will be created and stored.
            overwrite: (default = False) If overwrite is True then the memory will be overwrite any previous memory created in file. This parameter is applied only for file-based mode. 
        """

        if self.mode == 'nofile':
            self.addresses = SDM.create_addresses_DF(self.settings,random_seed= self.seed)
            self.counters = np.zeros((self.settings.numHL,(self.settings.max-self.settings.min+1),self.settings.dim))
        elif self.mode == 'parquet-hdf5':
            if not isinstance(file,str):
                raise TypeError("Wrong type 'file'.")
            
            self.file = file
            
            self.addresses = SDM.create_addresses_DF(self.settings,random_seed=self.seed)
            SDM.write_addresses(self.addresses, file+"addresses.parquet", 'overwrite' if overwrite else 'error')
            with h5py.File(file+"counters.hdf5", 'w') as countersF5:
                countersF5.attrs['min'] = self.settings.min
                countersF5.attrs['max'] = self.settings.max

                countersF5.attrs['numHL'] = self.settings.numHL
                countersF5.attrs['dim'] = self.settings.dim
                countersF5.attrs['radius'] = self.settings.radius
                countersF5.attrs['mode'] = self.mode
                ## counters for one hard location
                counter = np.zeros((self.settings.max-self.settings.min+1,self.settings.dim))
                #for each HL create a dataset with zeroes initialized
                for i in range(self.settings.numHL):
                    countersF5.create_dataset(str(i),data=counter)

        elif self.mode == 'hdf5':
            if not isinstance(file,str):
                raise TypeError("Wrong type 'file'.")
            random_seed = self.seed if self.seed != None else np.random.randint(0,100000)
            self.file = file
            with h5py.File(file+"counters.hdf5", 'w') as countersF5:
                countersF5.attrs['seed'] = random_seed
                countersF5.attrs['min'] = self.settings.min
                countersF5.attrs['max'] = self.settings.max
                countersF5.attrs['numHL'] = self.settings.numHL
                countersF5.attrs['dim'] = self.settings.dim
                countersF5.attrs['radius'] = self.settings.radius
                countersF5.attrs['mode'] = self.mode

                ## counters for one hard location
                counter = np.zeros((self.settings.max-self.settings.min+1,self.settings.dim))
                #for each HL create a dataset with zeroes initialized
                for i in range(self.settings.numHL):
                    countersF5.create_dataset(str(i),data=counter)


                        
    @staticmethod
    def open(sc,file):
        SDM.sc = sc
        #read hdf5 metadata and get the info of SDM then load the addresses to DF 
        spark = SparkSession.builder.appName("sdm").config(conf=SparkConf()).getOrCreate()
        with h5py.File(file+"counters.hdf5", 'r') as F5:
            settings = SDM.Settings(int(F5.attrs['min']),int(F5.attrs['max']),int(F5.attrs['dim']),\
                                int(F5.attrs['numHL']),int(F5.attrs['radius']))
            mode = F5.attrs['mode']
            
            sdm = SDM(sc,settings, mode = mode)
            random_seed = None
            sdm.file = file
            if mode == 'parquet-hdf5':
                sdm.addresses = spark.read.parquet(file+"addresses.parquet")

            elif mode == 'hdf5': 
                random_seed = int(F5.attrs['seed'])
                sdm.addresses = SDM.create_addresses_DF(settings,random_seed=random_seed)
        
        return sdm

    
    def write(self, word):
        assert len(word) == self.settings.dim
        activated = SDM.activated_hard_locations(self.addresses, word, self.settings.radius,self.settings.max)
        if self.mode == 'nofile':
            for index in activated:
                for i in range(len(word)):
                    self.counters[index][word[i]][i]=self.counters[index][word[i]][i]+1
        
        elif self.mode == 'parquet-hdf5' or self.mode == 'hdf5':
            with h5py.File(self.file+"counters.hdf5", 'r+') as countersF5:
                for index in activated:
                    dataset = countersF5[str(index)]
                    counter = dataset.value
                    for i in range(len(word)):
                        counter[word[i]][i]=counter[word[i]][i]+1
                    dataset[...] = counter

                    
    def read(self, word):
        assert len(word) == self.settings.dim
        activated = SDM.activated_hard_locations(self.addresses, word, self.settings.radius,self.settings.max)
        if self.mode == 'nofile':
            activatedCounters = list()
            for index in activated:
                activatedCounters.append(self.counters[index])
            
            return np.argmax(np.sum(activatedCounters,axis=0),axis=0)
        if self.mode == 'parquet-hdf5' or self.mode == 'hdf5':
            with h5py.File(self.file+"counters.hdf5", 'r') as countersF5:

                activatedCounters = list()
                for index in activated:
                    activatedCounters.append(countersF5[str(index)].value)

            return np.argmax(np.sum(activatedCounters,axis=0),axis=0)
