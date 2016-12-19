#!/usr/bin/env python2

import os, sys, glob, pickle
from os.path import join, isdir

# Make sure that you set this to the location your caffe2 library lies.
CAFFE2_ROOT = '/home/korsch/Repos/caffe2/'
OUTDIR = "tensors2"
sys.path.insert(0, join(CAFFE2_ROOT, 'gen'))

# After setting the caffe2 root path, we will import all the caffe2 libraries needed.
from caffe2.proto import caffe2_pb2
from caffe2.python import core, net_drawer, workspace, visualize, utils


# tensors contain all the parameters used in the net.
# The multibox model is relatively large so we have stored the parameters in multiple files.
file_parts = glob.glob("multibox_tensors.pb.part*")
file_parts.sort()
tensors = caffe2_pb2.TensorProtos()
tensors.ParseFromString(''.join(open(f).read() for f in file_parts))
if not isdir(OUTDIR):
	os.makedirs(OUTDIR)

for idx, param in enumerate(tensors.protos, 1):
	print "extracting {} / {} params...\r".format(idx, len(tensors.protos))
	if param.name != "nn0_w": continue
	import pdb; pdb.set_trace()
	# with open(join(OUTDIR, "{}.pkl".format(param.name)), "wb") as out:
	# 	pickle.dump(utils.Caffe2TensorToNumpyArray(param), out)

print

