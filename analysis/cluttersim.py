import sys
sys.path.insert(0, "../")
import src.io as io

cs = io.ClutterSim()
print(cs.N)

# do a single load and chop of one rsf file
dat = cs.extract(10, 0)
cs.set_max_dims(512, 4096)
cs.chop(dat, verbose=True)

# load all data
cs.load()

# move into tensorflow
tf = cs.tf_dataset(8)