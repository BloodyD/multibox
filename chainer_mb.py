#!/usr/bin/env python

import chainer, \
	chainer.links as L, \
	chainer.functions as F


clip = lambda val: F.clip(val, 0, 6)

class Mixed1(chainer.Chain):
	def __init__(self, pooling = "avg"):
		super(Mixed1, self).__init__(
			conv1x1 = L.Convolution2D(None, ..., ksize=1, stride=1, pad=2),	
			conv3x3_bottleneck = L.Convolution2D(None, ..., ksize=1, stride=1, pad=2),	
			conv3x3 = L.Convolution2D(None, ..., ksize=3, stride=1, pad=2),	
			conv3x3d_bottleneck = L.Convolution2D(None, ..., ksize=1, stride=1, pad=2),	
			conv3x3d_pre = L.Convolution2D(None, ..., ksize=3, stride=1, pad=2),	
			conv3x3d = L.Convolution2D(None, ..., ksize=3, stride=1, pad=2),	
			pool_reduce = L.Convolution2D(None, ..., ksize=1, stride=1, pad=2),	
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
	def __init__(self):
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
		super(Net, self).__init__(
			conv0 = L.Convolution2D(None, ..., kernel=7, stride=2, pad=2),
			conv1 = L.Convolution2D(None, ..., kernel=1, stride=1, pad=1),
			conv2 = L.Convolution2D(None, ..., kernel=3, stride=1, pad=2),
			
			mixed3a = Mixed1("avg"), 
			mixed3b = Mixed1("avg"), 
			mixed3c = Mixed2(), 

			mixed4a = Mixed1("avg"), 
			mixed4b = Mixed1("avg"), 
			mixed4c = Mixed1("avg"), 
			mixed4d = Mixed1("avg"), 
			mixed43 = Mixed2(), 
			
			mixed5a = Mixed1("avg"), 
			mixed5b = Mixed1("max"), 

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

