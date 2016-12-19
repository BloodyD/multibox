#!/usr/bin/env python

import chainer, \
	chainer.links as L, \
	chainer.functions as F

from os.path import join

clip = lambda val: F.clip(val, 0., 6.)

PADDING = 1

def pickled(transpose = None):
	def wrapper(func):
		def inner(*args, **kw):
			fname, folder = func(*args, **kw)
			data = join(folder, fname)
			try:
				with open(data, "rb") as f:
					res = pickle.load(f)
			except Exception:
				# in case the data is pickled with python2
				with open(data, "rb") as f:
					res = pickle.load(f, encoding = "latin1")

			if transpose is not None: res = res.transpose(transpose)
			return res
		return inner
	return wrapper

@pickled([0, 3, 1, 2])
def get_init_w(base, folder = "tensors"):
	return "{}_w.pkl".format(base), folder

def get_linear_init_w(base, folder = "tensors"):
	weights = get_init_w(base, folder)
	outs, channels, h, w = weights.shape
	return weights.reshape((outs, w * h * channels))

@pickled()
def get_init_b(base, folder = "tensors"):
	return "{}_b.pkl".format(base), folder


class Mixed1(chainer.Chain):
	def __init__(self, name, outputs, insize = None, pooling = "avg"):
		initialW = lambda key: get_init_w("{}_{}".format(name, key))
		initial_bias = lambda key: get_init_b("{}_{}".format(name, key))
		assert len(outputs) == 7, "there should be 7 output counts"
		super(Mixed1, self).__init__(
			conv1x1 = L.Convolution2D(insize, outputs[0],
				ksize=1, stride=1, pad=PADDING,
				initialW=initialW("1x1"), initial_bias=initial_bias("1x1")),

			conv3x3_bottleneck = L.Convolution2D(insize, outputs[1],
				ksize=1, stride=1, pad=PADDING,
				initialW=initialW("3x3_bottleneck"), initial_bias=initial_bias("3x3_bottleneck")),
			conv3x3 = L.Convolution2D(outputs[1], outputs[2],
				ksize=3, stride=1, pad=PADDING,
				initialW=initialW("3x3"), initial_bias=initial_bias("3x3")),

			conv3x3d_bottleneck = L.Convolution2D(insize, outputs[3],
				ksize=1, stride=1, pad=PADDING,
				initialW=initialW("3x3d_bottleneck"), initial_bias=initial_bias("3x3d_bottleneck")),
			conv3x3d_pre = L.Convolution2D(outputs[3], outputs[4],
				ksize=3, stride=1, pad=PADDING,
				initialW=initialW("3x3d_pre"), initial_bias=initial_bias("3x3d_pre")),
			conv3x3d = L.Convolution2D(outputs[4], outputs[5],
				ksize=3, stride=1, pad=PADDING,
				initialW=initialW("3x3d"), initial_bias=initial_bias("3x3d")),

			pool_reduce = L.Convolution2D(insize, outputs[6],
				ksize=1, stride=1, pad=PADDING,
				initialW=initialW("pool_reduce"), initial_bias=initial_bias("pool_reduce")),
		)
		self.pooling = F.average_pooling_2d if pooling == "avg" else F.max_pooling_2d

	def __call__(self, X):
		h1 = clip(self.conv1x1(X))

		h2 = clip(self.conv3x3_bottleneck(X))
		h2 = clip(self.conv3x3(h2))

		h3 = clip(self.conv3x3d_bottleneck(X))
		h3 = clip(self.conv3x3d_pre(h3))
		h3 = clip(self.conv3x3d(h3))

		h4 = self.pooling(X, ksize=3, stride=1, pad=PADDING)
		h4 = clip(self.pool_reduce(h4))

		try:
			return F.concat([h1, h2, h3, h4], axis=1)
		except Exception as e:
			import pdb; pdb.set_trace()
			raise e

class Mixed2(chainer.Chain):
	def __init__(self, name, outputs):
		assert len(outputs) == 5, "there should be 5 output counts"
		initialW = lambda key: get_init_w("{}_{}".format(name, key))
		initial_bias = lambda key: get_init_b("{}_{}".format(name, key))
		super(Mixed2, self).__init__(
			conv3x3_bottleneck = L.Convolution2D(None, outputs[0],
				ksize=1, stride=1, pad=PADDING,
				initialW=initialW("3x3_bottleneck"), initial_bias=initial_bias("3x3_bottleneck")),
			conv3x3 = L.Convolution2D(outputs[0], outputs[1],
				ksize=3, stride=2, pad=PADDING,
				initialW=initialW("3x3"), initial_bias=initial_bias("3x3")),

			conv3x3d_bottleneck = L.Convolution2D(None, outputs[2],
				ksize=1, stride=1, pad=PADDING,
				initialW=initialW("3x3d_bottleneck"), initial_bias=initial_bias("3x3d_bottleneck")),
			conv3x3d_pre = L.Convolution2D(outputs[2], outputs[3],
				ksize=3, stride=1, pad=PADDING,
				initialW=initialW("3x3d_pre"), initial_bias=initial_bias("3x3d_pre")),
			conv3x3d = L.Convolution2D(outputs[3], outputs[4],
				ksize=3, stride=2, pad=PADDING,
				initialW=initialW("3x3d"), initial_bias=initial_bias("3x3d")),
		)

	def __call__(self, X):

		h1 = clip(self.conv3x3_bottleneck(X))
		h1 = clip(self.conv3x3(h1))

		h2 = clip(self.conv3x3d_bottleneck(X))
		h2 = clip(self.conv3x3d_pre(h2))
		h2 = clip(self.conv3x3d(h2))

		h3 = F.max_pooling_2d(X, ksize=3, stride=2, pad=PADDING)

		try:
			logging.debug([h.shape for h in [h1, h2, h3]])
			return F.concat([h1, h2, h3], axis=1)
		except Exception as e:
			import pdb; pdb.set_trace()
			raise e


class Net(chainer.Chain):
	def __init__(self):
		super(Net, self).__init__(
			conv0 = L.Convolution2D(3, 64, ksize=7, stride=2, pad=PADDING,
				initialW=get_init_w("conv2d0"), initial_bias=get_init_b("conv2d0")),
			conv1 = L.Convolution2D(64, 64, ksize=1, stride=1, pad=1,
				initialW=get_init_w("conv2d1"), initial_bias=get_init_b("conv2d1")),
			conv2 = L.Convolution2D(64, 192, ksize=3, stride=1, pad=PADDING,
				initialW=get_init_w("conv2d2"), initial_bias=get_init_b("conv2d2")),

			mixed3a = Mixed1(name="mixed3a", insize = 192,
				outputs=[64,  64,  64, 64, 96, 96, 32], pooling="avg"),
			mixed3b = Mixed1(name="mixed3b", insize = 256,
				outputs=[64,  64,  96, 64, 96, 96, 64], pooling="avg"),
			mixed3c = Mixed2(name="mixed3c",
				outputs=[128, 160, 64, 96, 96]),

			mixed4a = Mixed1(name="mixed4a",
				outputs=[224, 64,  96,  96,  128, 128, 128], pooling="avg"),
			mixed4b = Mixed1(name="mixed4b",
				outputs=[192, 96,  128, 96,  128, 128, 128], pooling="avg"),
			mixed4c = Mixed1(name="mixed4c",
				outputs=[160, 128, 160, 128, 160, 160, 96], pooling="avg"),
			mixed4d = Mixed1(name="mixed4d",
				outputs=[96,  128, 192, 160, 192, 192, 96], pooling="avg"),
			mixed4e = Mixed2(name="mixed4e",
				outputs=[128, 192, 192, 256, 256]),

			mixed5a = Mixed1(name="mixed5a",
				outputs=[352, 192, 320, 160, 224, 224, 128], pooling="avg"),
			mixed5b = Mixed1(name="mixed5b",
				outputs=[352, 192, 320, 192, 224, 224, 128], pooling="max"),

			fc6 = L.Linear(1024 * 3 * 3, 4096,
				initialW=get_linear_init_w("nn0"), initial_bias=get_init_b("nn0")),
			fc7 = L.Linear(4096, 4096,
				initialW=get_linear_init_w("nn1"), initial_bias=get_init_b("nn1")),

			location = L.Linear(4096, 3200,
				initialW=get_linear_init_w("imagenet_location_projection"),
				initial_bias=get_init_b("imagenet_location_projection")),

			confidence = L.Linear(4096, 800,
				initialW=get_linear_init_w("imagenet_confidence_projection"),
				initial_bias=get_init_b("imagenet_confidence_projection")),
		)

	def __call__(self, X):
		logging.debug(("init", X.shape))

		h = clip(self.conv0(X))
		h = F.max_pooling_2d(h, ksize=3, stride=2, pad=PADDING)
		h =  clip(self.conv1(h))
		h =  clip(self.conv2(h))
		h = F.max_pooling_2d(h, ksize=3, stride=2, pad=PADDING)
		logging.debug(("pre", h.shape))

		h = self.mixed3a(h)
		logging.debug(("3a", h.shape))
		h = self.mixed3b(h)
		logging.debug(("3b", h.shape))
		h = self.mixed3c(h)
		logging.debug(("3c", h.shape))

		h = self.mixed4a(h)
		logging.debug(("4a", h.shape))
		h = self.mixed4b(h)
		logging.debug(("4b", h.shape))
		h = self.mixed4c(h)
		logging.debug(("4c", h.shape))
		h = self.mixed4d(h)
		logging.debug(("4d", h.shape))
		h = self.mixed4e(h)
		logging.debug(("4e", h.shape))

		h = self.mixed5a(h)
		logging.debug(("5a", h.shape))
		h = self.mixed5b(h)
		logging.debug(("5b", h.shape))

		h = F.average_pooling_2d(h, ksize=3, stride=6, pad=1)
		# h = F.max_pooling_2d(h, ksize=3, stride=3, pad=1)
		logging.debug(("post", h.shape))
		h = clip(self.fc6(h))
		h = clip(self.fc7(h))

		return self.location(h), self.confidence(h)



import glob, pickle, simplejson as json
import logging, numpy as np
from skimage import io, transform
from matplotlib import pyplot as plt

DEBUG = 0

logging.basicConfig(
	format = '%(levelname)s - [%(asctime)s] %(filename)s:%(lineno)d [%(funcName)s]: %(message)s',
	level = logging.INFO if DEBUG else logging.INFO)


def PrintBox(loc, height, width, style='r-'):
    """A utility function to help visualizing boxes."""
    x1,y1,x2,y2 = loc
    xmin, ymin, xmax, ymax = x1 * width, y1 * height, x2 * width, y2 * height
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], style)

def main(image_names):
	logging.info("loading net...")
	net = Net()
	logging.info("net loaded")
	for image_name in image_names:
		image = transform.resize(io.imread(image_name), (SIZE, SIZE))
		data = image[np.newaxis].astype(np.float32).transpose((0, 3, 1, 2))
		data -= 0.5
		var = chainer.Variable(data)

		location, confidence = net(var)
		LOCATION_PRIOR = np.loadtxt('ipriors800.txt')
		location = location.data * LOCATION_PRIOR[:,0] + LOCATION_PRIOR[:,1]
		location = location.reshape((800, 4))

		img = io.imread(image_name)
		plt.imshow(img)
		plt.axis("off")
		# Let's show the most confident 5 predictions.
		# Note that argsort sorts things in increasing order.
		sorted_idx = np.argsort(confidence.data[0])
		for idx in sorted_idx[-5:]:
			# import pdb; pdb.set_trace()
			# print(idx)
			PrintBox(location[idx], img.shape[0], img.shape[1])

		plt.show()

SIZE = 224
if __name__ == '__main__':
	main(glob.glob("*.jpg"))
