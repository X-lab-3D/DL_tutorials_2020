import os
import torch
import imageio
import torch.nn as nn
import torch.nn.functional as F


# Assignment 1a
def createCoordinates(imageShape):
	x = torch.range(0, imageShape[1]-1).repeat(imageShape[0], 1).unsqueeze(0) / (imageShape[1]-1)
	y = torch.range(0, imageShape[0]-1).repeat(imageShape[1], 1).T.unsqueeze(0) / (imageShape[0]-1)
	coordinates = torch.cat((x,y)).permute(1,2,0).view(-1,2)
	return coordinates
	
# Assignment 1b
def photo2dataset(imageLocation):
	image = (torch.Tensor(imageio.imread(imageLocation)[:,:,:3])/255.)
	coordinates = createCoordinates(image.shape)
	labels = image.view(-1,3)
	shape = image.shape
	return shape, coordinates, labels
	
# Assignment 2a
class FourierModule(nn.Module):
	def __init__(self, dimension, scale):
		super(FourierModule, self).__init__()
		self.B = torch.randn(2, dimension)*6.28*scale
	def forward(self, v):
		return torch.cat((
				torch.sin(torch.matmul(v, self.B)), 
				torch.cos(torch.matmul(v, self.B))), -1)

# Assignment 2b
def assignment2b(dimension, scale):
	# Create the module
	module = FourierModule(dimension, scale)
	# Make a folder to put results in
	fname = 'assignment2b_dimension=%i_scale=%i' % (dimension, scale)
	try: os.mkdir(fname)
	except: pass
	# Create the coordinates
	coordinates = createCoordinates((100,100))
	# Create the output
	output = module(coordinates)
	# Save the outputs
	for i in range(dimension):
		imageio.imsave('%s/%i.png' % (fname,i), output[:,i].reshape(100,100))

# Assignment 3a
class Model(nn.Module):
	def __init__(self, dimension, scale):
		super(Model, self).__init__()
		self.net = nn.Sequential(
						FourierModule(dimension, scale),
						nn.Linear(dimension*2, 256), nn.ReLU(),
						nn.Linear(256, 256), nn.ReLU(),
						nn.Linear(256, 3), nn.Sigmoid(),)
	def forward(self, x):
		return self.net(x)
		
class Model_not_fourier(nn.Module):
	def __init__(self):
		super(Model_not_fourier, self).__init__()
		self.net = nn.Sequential(
						nn.Linear(2, 256), nn.ReLU(),
						nn.Linear(256, 256), nn.ReLU(),
						nn.Linear(256, 3), nn.Sigmoid(),)
	def forward(self, x):
		return self.net(x)

# Assignment 3b	
def trainOneEpoch(model, coordinates, labels, mb=128):
	p = torch.randperm(len(coordinates))
	coordinates = coordinates[p]
	labels = labels[p]
	total = 0
	optim = torch.optim.Adam(model.parameters())
	for i in range(0, len(coordinates), mb):
		batch = coordinates[i:i+mb]
		labs  = labels[i:i+mb]
		prediction = model(batch)
		loss = F.binary_cross_entropy(prediction, labs, reduction='sum')
		loss.backward()
		optim.step()
		optim.zero_grad()
		total += float(loss.detach())
	return total

def saveTestImage(model, coordinates, image_shape, output_name):
	with torch.no_grad():
		prediction = model(coordinates)
		prediction = prediction.view(image_shape)
		prediction = prediction*255
		prediction = prediction.numpy().astype('uint8')
		imageio.imsave(output_name, prediction)

# Assignment 3
def train_and_test(image_location, dimension, scale, epochs, outputName, new_shape=None, model='fourier'):
	shape, coordinates, labels  = photo2dataset(image_location)
	if new_shape==None: new_shape=shape
	new_coordinates = createCoordinates(new_shape)
	if model == 'fourier':
		net = Model(dimension, scale)
	else:
		net = Model_not_fourier()
	optim = torch.optim.Adam(net.parameters())
	for epoch in range(epochs):
		loss = trainOneEpoch(net, coordinates, labels)
		print (epoch, loss)
	saveTestImage(net, new_coordinates, (new_shape[0], new_shape[1],3), outputName)

if __name__ == "__main__":
	# Hier kunnen jullie de assignments een voor een uitproberen. 
	# Uncomment de assignment die je wil testen en vul de juiste waardens in. 
	# Succes!
	###  Assignment 2b: 
	#assignment2b(dimension, scale)
	#assignment2b(dimension, scale)
	#Question: How does the scale parameter affect the outputs?
	
	### Assignment 3c (outputName should be something like 'output.png')
	#train_and_test(image_location, dimension, scale, epochs, outputName)
	
	### Assignment 3d 
	#train_and_test(image_location, dimension, scale, epochs, outputName)

	### Assignment 3e 
	#train_and_test(image_location, dimension, scale, epochs, outputName, new_image_shape)
	#train_and_test(image_location, dimension, scale, epochs, outputName, new_image_shape)
	#train_and_test(image_location, dimension, scale, epochs, outputName, new_image_shape)
	#train_and_test(image_location, dimension, scale, epochs, outputName, new_image_shape)
	
	### Assignment 3f 
	#train_and_test(image_location, dimension, scale, epochs, 'met_fourier.png')
	#train_and_test(image_location, dimension, scale, epochs, 'zonder_fourier.png', model='no_fourier')
	
	
	
	
		
