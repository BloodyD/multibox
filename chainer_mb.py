#!/usr/bin/env python

import chainer, \
	chainer.links as L, \
	chainer.functions as F


clip = lambda val: F.clip(val, 0, 6)

class Mixed1(chainer.Chain):
	def __init__(self, outputs, pooling = "avg"):
		assert len(outputs) == 7, "there should be 7 output counts"
		super(Mixed1, self).__init__(
			conv1x1 = L.Convolution2D(None, outputs[0], ksize=1, stride=1, pad=2),

			conv3x3_bottleneck = L.Convolution2D(None, outputs[1], ksize=1, stride=1, pad=2),
			conv3x3 = L.Convolution2D(None, outputs[2], ksize=3, stride=1, pad=2),

			conv3x3d_bottleneck = L.Convolution2D(None, outputs[3], ksize=1, stride=1, pad=2),
			conv3x3d_pre = L.Convolution2D(None, outputs[4], ksize=3, stride=1, pad=2),
			conv3x3d = L.Convolution2D(None, outputs[5], ksize=3, stride=1, pad=2),

			pool_reduce = L.Convolution2D(None, outputs[6], ksize=1, stride=1, pad=2),
		)
		self.pooling = F.average_pooling_2d if pooling == "avg" else F.max_pooling_2d

	def __call__(self, X):
		h1 = clip(self.conv1x1(X))

		h2 = clip(self.conv3x3_bottleneck(X))
		h2 = clip(self.conv3x3(h2))

		h3 = clip(self.conv3x3d_bottleneck(X))
		h3 = clip(self.conv3x3d_pre(h3))
		h3 = clip(self.conv3x3d(h3))

		h4 = self.pooling(X, ksize=3, stride=1, pad=2)
		h4 = clip(self.pool_reduce(X))

		import pdb; pdb.set_trace()
		# todo: DepthConcat
		return [h1, h2, h3, h4]

class Mixed2(chainer.Chain):
	def __init__(self, outputs):
		super(Mixed2, self).__init__(
			conv3x3_bottleneck = L.Convolution2D(None, ..., ksize=1, stride=1, pad=2),
			conv3x3 = L.Convolution2D(None, ..., ksize=3, stride=2, pad=2),
			conv3x3d_bottleneck = L.Convolution2D(None, ..., ksize=1, stride=1, pad=2),
			conv3x3d_pre = L.Convolution2D(None, ..., ksize=3, stride=1, pad=2),
			conv3x3d = L.Convolution2D(None, ..., ksize=3, stride=2, pad=2),
		)

	def __call__(self, X):

		h1 = clip(self.conv3x3_bottleneck(X))
		h1 = clip(self.conv3x3(h1))

		h2 = clip(self.conv3x3d_bottleneck(X))
		h2 = clip(self.conv3x3d_pre(h2))
		h2 = clip(self.conv3x3d(h2))

		h3 = F.max_pooling_2d(X, ksize=3, stride=2, pad=2)

		import pdb; pdb.set_trace()
		# todo: DepthConcat
		return [h1, h2, h3]

class Net(chainer.Chain):
	def __init__(self):
		outputs =
		super(Net, self).__init__(
			conv0 = L.Convolution2D(None, 64, kernel=7, stride=2, pad=2),
			conv1 = L.Convolution2D(None, 64, kernel=1, stride=1, pad=1),
			conv2 = L.Convolution2D(None, 192, kernel=3, stride=1, pad=2),

			mixed3a = Mixed1(outputs=[64, 64, 64, 64, 96, 96, 32], pooling="avg"),
			mixed3b = Mixed1(outputs=[64, 64, 96, 64, 96, 96, 64], pooling="avg"),
			mixed3c = Mixed2(outputs=[]),

			mixed4a = Mixed1(outputs=[224, 64,  96,  96,  128, 128, 128], pooling="avg"),
			mixed4b = Mixed1(outputs=[192, 96,  128, 96,  128, 128, 128], pooling="avg"),
			mixed4c = Mixed1(outputs=[160, 128, 160, 128, 160, 160, 96], pooling="avg"),
			mixed4d = Mixed1(outputs=[96,  128, 192, 160, 192, 192, 96], pooling="avg"),
			mixed4e = Mixed2(outputs=[]),

			mixed5a = Mixed1(outputs=[352, 192, 320, 160, 224, 224, 128], pooling="avg"),
			mixed5b = Mixed1(outputs=[352, 192, 320, 192, 224, 224, 128], pooling="max"),

			fc6 = L.Linear(None, ...),
			fc7 = L.Linear(None, ...),

			location = L.Linear(None, ...),
			confidence = L.Linear(None, ...),
		)

	def __call__(self, X):
		h = clip(self.conv0(X))
		h = F.max_pooling_2d(h, ksize=3, stride=2, pad=2)
		h =  clip(self.conv1(h, ksize=1, stride=1, pad=1))
		h =  clip(self.conv2(h, ksize=3, stride=1, pad=2))
		h = F.max_pooling_2d(h, ksize=3, stride=2, pad=2)

		h = self.mixed3a(h)
		h = self.mixed3b(h)
		h = self.mixed3c(h)

		h = self.mixed4a(h)
		h = self.mixed4b(h)
		h = self.mixed4c(h)
		h = self.mixed4d(h)
		h = self.mixed4e(h)

		h = self.mixed5a(h)
		h = self.mixed5b(h)

		h = F.average_pooling_2d(h, ksize=3, stride=2, pad=1)
		h = clip(self.fc6(h))
		h = clip(self.fc7(h))

		return self.confidence(h), self.confidence(location)



import glob, pickle, simplejson as json

if __name__ == '__main__':
	res = {}
	for fname in glob.glob("tensors/*.pkl"):
		with open(fname, "rb") as f:
			tensor = pickle.load(f, encoding="latin1")
			if tensor.ndim == 4:
				n_outs, h, w, channels = tensor.shape
				res[fname] = dict(outputs=n_outs, h=h, w=w, channels=channels)
			else:
				res[fname] = dict(channels=tensor.shape[0])

	with open("layer_info.json", "w") as out:
		json.dump(res, out, indent = 2)
