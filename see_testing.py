import src.io as io

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

cs = io.carrierset()
cs.show()