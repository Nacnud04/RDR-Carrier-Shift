#!/usr/bin/env python

import rsf.api as rsf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def save_rsf_2D(data, path, d1=None, d2=None, o1=None, o2=None,
                label1=None, label2=None, unit1=None, unit2=None, f=None):
    
    """
    Export a 2D numpy array as a .rsf file. Axis deltas, origins, labels
    and units can be individually specified. If a existing file path is  
    provided (f) the exported file will use the header from that file to
    define the axes.  
    """
    
    Fo = rsf.Output(path)

    Fo.put("n1", data.shape[1])
    Fo.put("n2", data.shape[0])

    if d1:
        Fo.put("d1", d1)
    if d2:
        Fo.put("d2", d2)
    if o1:
        Fo.put("o1", o1)
    if o2:
        Fo.put("o2", o2)
    if label1:
        Fo.put("label1", label1)
    if label2:
        Fo.put("label2", label2)
    if unit1:
        Fo.put("unit1", unit1)
    if unit2:
        Fo.put("unit2", unit2)

    # if an existing rsf file is given
    # use the axis from that file on the exported file
    if f:

        f = rsf.Input(f)

        # dim 1
        Fo.put("n1", f.int('n1'))
        Fo.put("d1", f.float('d1'))
        Fo.put("o1", f.float('o1'))
        Fo.put("label1",str(f.string('label1')))
        Fo.put("unit1",str(f.string('unit1')))

        # dim 2
        Fo.put("n2", f.int('n2'))
        Fo.put("d2", f.float('d2'))
        Fo.put("o2", f.float('o2'))
        Fo.put("label2",str(f.string('label2')))
        Fo.put("unit2",str(f.string('unit2')))

    print(data.dtype)

    Fo.write(data)

    Fo.close()

# class which quickly reads in rsf files into numpy arrays
class rsffile():

    def __init__(self, path, dtype='float32'):

        f = rsf.Input(path)
        
        # space domain stuff
        self.nx = f.int('n1')
        self.dx = f.float('d1')
        self.ox = f.float('o1')
        self.lx = f.string('label1')
        self.ux = f.string('unit1')

        # time domain stuff
        self.nt = f.int('n2')
        self.dt = f.float('d2')
        self.ot = f.float('o2')
        self.lt = f.string('label2')
        self.ut = f.string('unit2')

        if f.int('n3'):

            # other axis which is a bunch of different examples stacked on each other (in this instance)
            self.nc = f.int('n3')

            # form empty array to house data
            self.amps = np.zeros((self.nc, self.nt, self.nx, 1), dtype=dtype)

        else:

            self.amps = np.zeros((self.nt, self.nx), dtype=dtype)

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

    _modules_imported = False

    def __new__(cls, *args, **kwargs):
        if not cls._modules_imported:
            cls._perform_imports()
            cls._modules_imported = True
        return super(dataset, cls).__new__(cls)

    @classmethod
    def _perform_imports(cls):
        global tf_data, tf
        from tensorflow import data as tf_data
        import tensorflow as tf

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
    
import glob

class ClutterSim():

    _modules_imported = False

    def __new__(cls, *args, **kwargs):
        if not cls._modules_imported:
            cls._perform_imports()
            cls._modules_imported = True
        return super(ClutterSim, cls).__new__(cls)

    @classmethod
    def _perform_imports(cls):
        global tf_data, tf
        from tensorflow import data as tf_data
        import tensorflow as tf

    def __init__(self, dir="/home/byrne/WORK/research/mars2024/mltrSPSLAKE/T", nc=3, 
            maxfiles=1500, cropfilelist=False, combinerandom=False):

        self.random = combinerandom

        # generate list of rsf files in each carrier set
        self.cs = [np.sort(glob.glob(f"{dir}/dsyT{c}-r*.rsf")) for c in range(nc)]

        for c, j in zip(self.cs, range(nc)):
            print(f"Found {len(c)} realizations of carrier {j}")

        self.ids = [[int(f.split("-r")[1].split(".")[0]) for f in c] for c in self.cs]
        self.ids0 = np.array(self.ids[0])
        self.ids2 = np.array(self.ids[2])

        self.N = len(self.cs[2])

        # max file size
        maxfiles = maxfiles
        if maxfiles < self.N: 
            self.N = maxfiles
        print(f'n: {self.N}')

        # crop file list (this can cause issues if files are not in order
        # and the maxfile count is less than the total filecount)
        if cropfilelist:
            for i in range(nc):
                self.cs[i] = self.cs[i][:maxfiles]

    def extract(self, i, carrier):

        # pull correct file out of list of rsf files
        if carrier == 0:
            path = self.cs[carrier][np.where(self.ids0 == i)][0]
        elif carrier == 2:
            path = self.cs[carrier][np.where(self.ids2 == i)][0]

        # turn rsf file into rsffile object to grab data
        file = rsffile(path)

        return file.amps
    
    def grab_pair(self, i):

        pair = self.pair

        if len(pair) != 2:
            raise IndexError(f"A pair can only be of length 2. Received length of: {len(pair)}")
        
        A = self.extract(i, pair[0])
        B = self.extract(i, pair[1])

        return A, B
    
    def set_max_dims(self, nx, nt):

        self.maxx = nx # max amount of possible indicies along x
        self.maxt = nt # max amount of possible indicies along y

        self.img_size = (self.maxx, self.maxt)

    def chop(self, array, verbose=False):
        
        nt = array.shape[1]
        nx = array.shape[0]

        self.fullnt = nt
        self.fullnx = nx

        if self.maxt < nt:
            raise NotImplementedError(f"Time axis too large {self.maxt} < {nt}. \
                                      Cannot chop radargram by along time axis")
        
        elif self.maxx > nx:
            return [array]
        
        sections = 1 + (nx // self.maxx) # how many sections do we divide the radargram into?
        whitespace = 1 + self.maxx - (nx % self.maxx) # how much 0 padding in the x dimension for the last slice?

        self.sections = sections
        self.whitespace = whitespace

        if verbose:
            print(f"Sections: {sections} | Whitespace: {whitespace}")

        slices = []

        for s in range(sections):

            # start and end of x direction
            if s == sections - 1: # last section
                st, en = s * self.maxx, -1 
            else: 
                st, en = s * self.maxx, (s + 1) * self.maxx

            if verbose:
                print(f"st: {st} | en: {en}")

            slice = array[st:en, :] # chop array

            # set max as one
            #scrslice /= np.max(slice)

            # pad along x
            if s == sections - 1:
                slice = np.concatenate((slice, np.zeros((whitespace, nt))), axis=0)

            # pad along t
            slice = np.concatenate((slice, np.zeros((self.maxx, self.maxt-nt))), axis=1)

            slices.append(slice)

        return slices
    

    def decompose(self, array):

        nt = array.shape[1]
        nx = array.shape[0]

        self.fullnt = nt
        self.fullnx = nx

        self.sections = nx

        if self.maxt < nt:
            raise NotImplementedError(f"Time axis too large {self.maxt} < {nt}. \
                                      Cannot chop radargram by along time axis")
        
        outputs = []
        for a in list(array):
            outputs.append(a.reshape((1, 512)))

        return outputs


    def stitch(self, predictions):

        # how many arrays in the output?
        count = len(predictions) // self.sections

        print(f"Stitching into {count} radargrams")

        rdrgrms = []

        # iterate over output images
        for i in range(count):

            if self.maxx != 1:

                # grab all slices of predicted image to stitch together
                slices = [predictions[j+i*self.sections] for j in range(self.sections)]

                # stitch
                output = np.vstack(tuple(slices))

                # chop off whitespace
                rdrgrms.append(output[:self.fullnx, :self.fullnt])
            
            else:

                # grab all slices of predicted image to stitch together
                slices = [predictions[j+i*self.sections] for j in range(self.sections)]

                # stitch
                output = np.vstack(slices)
                
                # split each slice/batch into traces

                rdrgrms.append(output)

        return rdrgrms

    
    # load all data into memory and chop it up
    # that way it can be easily loaded into tf
    def load(self, pair=(0, 2)):
        
        self.pair = pair

        self.As = [] # source data
        self.Bs = [] # target data

        for n in range(self.N):
            
            print(f"Loading... {n+1}/{self.N}", end="   \r")

            A, B = self.grab_pair(n) # grab data

            print(f"Chopping... {n+1}/{self.N}", end="   \r")

            if self.maxx > 1:
                A = self.chop(A) # chop up A and B
                B = self.chop(B) # chop up A and B

            elif self.maxx == 1:
                # decompose into individual traces as input
                A = self.decompose(A)
                B = self.decompose(B)

            else:
                raise ValueError(f"Maximum number of traces (maxx/nx) must be \
                                 >= 1. The received input was {self.maxx}")

            # add to data arrays
            self.As.extend(A)
            self.Bs.extend(B)

        print("\n")

    
    # --- TENSORFLOW FUNCTIONS ---
    # make pair tf compatable

    def grab_pair_tf(self, i):

        if self.random:

            # update
            print(f"Combining random {i}", end="     \r")
            # how many to randomly sum?
            n = 7 
            # seed based on i
            np.random.seed(i)
            # get indicies of things to sum
            sumids = np.random.randint(0, self.lentrain, size=n)
            # set initial array
            A, B = np.copy(self.As[sumids[0]]), np.copy(self.Bs[sumids[0]])
            # iterate through others
            for j in range(1, n):
                idx = self.train_slices[sumids[j]]
                # detemine roll direction and random flip
                fac = 1
                if j % 2 == 0:
                    fac = -1
                    self.As[idx] = np.flip(self.As[idx], axis=0)
                    self.Bs[idx] = np.flip(self.Bs[idx], axis=0)
                # sum and roll
                A += np.roll(self.As[idx], fac * self.maxt // n, axis=1)
                B += np.roll(self.Bs[idx], fac * self.maxt // n, axis=1)

        else:

            # turn i into a real number from tensorflow notation
            i = tf.cast(i, tf.int32).numpy()
            # grab pair
            A, B = self.As[i], self.Bs[i]
            # convert from numpy to tensorflow
        
        return tf.convert_to_tensor(A), tf.convert_to_tensor(B)
    
    def tf_dataset(self, batch_size, testing, bitdepth=64, randomtrainsize=None):

        def _set_shapes(A, B):
            A.set_shape(self.img_size)
            B.set_shape(self.img_size)
            return A, B
        
        # set bitdepth
        if bitdepth == 64:
            bd = tf.float64
        elif bitdepth == 32:
            bd = tf.float32
        
        # create list of testing indicies
        seed = 2024
        np.random.seed(seed)
        testlen = int(self.N * testing)
        testing_ids = np.random.randint(0, self.N, size=testlen)
        self.testids = testing_ids
        training_ids = [i for i in range(self.N) if i not in testing_ids]
        self.trainids = training_ids

        # convert lists to tensors for tf
        test_slices = np.array([np.arange(self.sections) + i * self.sections for i in testing_ids]).flatten()
        self.lentest = len(test_slices)
        print(f"Test input: {len(test_slices)}")
        test_ids = tf.convert_to_tensor(test_slices)

        train_slices = np.array([np.arange(self.sections) + i * self.sections for i in training_ids]).flatten()
        self.train_slices = train_slices
        self.lentrain = len(train_slices)
        print(f"Train input: {len(train_slices)}")
        if not randomtrainsize:
            train_ids = tf.convert_to_tensor(train_slices)
        else:
            train_slices = np.arange(randomtrainsize * self.sections)
            train_ids = tf.convert_to_tensor(train_slices)

        # --- LOAD TESTING DATA ---

        indices = tf.data.Dataset.from_tensor_slices(test_ids)

        tfdataset = indices.map(lambda i: tf.py_function(
                                func=self.grab_pair_tf,
                                inp=[i],
                                Tout=(bd, bd)),  
                                num_parallel_calls=tf_data.AUTOTUNE)
        
        
        tfdataset = tfdataset.map(_set_shapes, num_parallel_calls=tf.data.AUTOTUNE)
        
        test = tfdataset.batch(batch_size, drop_remainder=True)

        # --- LOAD TRAINING DATA ---

        print(f"Train input cluttersims: {len(train_slices) // self.sections}")
        indices = tf.data.Dataset.from_tensor_slices(train_ids)

        tfdataset = indices.map(lambda i: tf.py_function(
                                func=self.grab_pair_tf,
                                inp=[i],
                                Tout=(bd, bd)),  
                                num_parallel_calls=tf_data.AUTOTUNE)
        
        
        tfdataset = tfdataset.map(_set_shapes, num_parallel_calls=tf.data.AUTOTUNE)
        
        train = tfdataset.batch(batch_size, drop_remainder=True)

        return train, test
