#!/usr/bin/env python

import rsf.api as rsf
import numpy as np
import matplotlib.pyplot as plt
import time

import keras
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
import tensorflow as tf

# class which quickly reads in rsf files into numpy arrays
class rsffile():

    def __init__(self, path):

        f = rsf.Input(path)
        
        # time domain stuff
        self.nt = f.int('n1')
        self.dt = f.float('d1')
        self.ot = f.float('o1')
        self.lt = f.string('label1')
        self.ut = f.string('unit1')

        # space domain stuff
        self.nx = f.int('n2')
        self.dx = f.float('d2')
        self.ox = f.float('o2')
        self.lx = f.string('label2')
        self.ux = f.string('unit2')

        # other axis which is a bunch of different examples stacked on each other
        self.nc = f.int('n3')

        # develop empty array to house data
        self.amps = np.zeros((self.nc, self.nx, self.nt, 1), dtype='float32')

        # load in data
        f.read(self.amps)

        self.path = path

        f.close()

    def show(self, i):

        plt.imshow(self.amps[i,:,:], cmap="gray")
        plt.show()


# class to see a set of different frequencies
class carrierset():

    def __init__(self, dir="random_planes", testing=False):

        carriers = (0, 1, 2)
        self.titles = ("High", "Med", "Low")

        self.files = []
        self.paths = []

        for c in carriers:

            f = rsffile(f"{dir}/dat{c}.rsf")
            self.files.append(f)
            self.paths.append(f"{dir}/dat{c}.rsf")
            self.count = f.nc

    def show(self, i=10):

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, f, t in zip(axes, self.files, self.titles):
            ax.imshow(f.amps[i,:,:], cmap="gray",
                      extent=[f.ox,f.ox+f.dx*f.nx,f.ot+f.dt*f.nt,f.ot],
                      aspect='auto')
            ax.set_title(t)
            ax.set_xlabel("x (.)")
            ax.set_ylabel(r"t ($\mu$)")

        plt.suptitle(i)
        plt.show()

    def grab_pair(self, i, pair=(0, 1)):
        return self.files[pair[0]].amps[i,:,:], self.files[pair[1]].amps[i,:,:]
    
    def grab_pair_paths(self, pair=(0, 1)):
        return self.paths[pair[0]], self.paths[pair[1]]
    

class dataset():

    def __init__(self, dir, pair=(0, 1)):

        self.cs = carrierset(dir=dir)
        self.pair = pair

    def export_all(self, i):
        
        realA, realB = [], []

        A, B = self.cs.grab_pair(i)
        realA.append(A)
        realB.append(B)

        return np.array([realA, realB])
    
    def export_carrierset(self, i):
        return self.cs.grab_pair(i, self.pair)
    
    def export_carrierset_wrapper(self, i):
        i = tf.cast(i, tf.int32).numpy()  # Convert Tensor to numpy integer
        A, B = self.export_carrierset(i)
        return tf.convert_to_tensor(A), tf.convert_to_tensor(B)

    def show_random(self, n=3):
        for i in range(n):
            x = np.random.randint(0, self.cs.count, 1)[0]
            self.cs.show(x)

    def tf_dataset(self, batch_size, img_size):

        def _set_shapes(A, B):
            A.set_shape(img_size)
            B.set_shape(img_size)
            return A, B

        indices = tf.data.Dataset.from_tensor_slices(tf.range(self.cs.count))

        tfdataset = indices.map(lambda i: tf.py_function(
                                func=self.export_carrierset_wrapper,
                                inp=[i],
                                Tout=(tf.float32, tf.float32)),  
                                num_parallel_calls=tf_data.AUTOTUNE)
        
        
        tfdataset = tfdataset.map(_set_shapes, num_parallel_calls=tf.data.AUTOTUNE)
        
        return tfdataset.batch(batch_size)
    