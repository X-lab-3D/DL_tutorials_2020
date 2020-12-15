import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

# for REPRODUCIBILITY
torch.manual_seed(1234)

pdb1 = '1zs8.pdb' # Murine MHC Class Ib Molecule M10.5
pdb2 = '4l3c.pdb' # HLA-A2 in complex with D76N b2m mutant

# Extracting xyz coordinates and aa-index from C-alpha's of chain A
# aa-indices are removed later
def extract(pdbFile): 
	pdb_ca_coordinates = [[line[30:38], line[38:46], line[46:54], line[23:26]] 
				   for line in open(pdbFile).read().split('\n')
				   if line.startswith('ATOM ') 
				   and line[21] == 'A'
				   and line[13:15] == 'CA']
	return [[float(item) for item in vec]for vec in pdb_ca_coordinates]

# Saves coordinates in a file for pymol
# (can be read by associated PyMOL plugin)
def save(fileName, coordinates):
	newFile = open(fileName, 'w')
	[newFile.write(' '.join([str(number) 
				   for number in coordinate])+'\n')
				   for coordinate in coordinates]
	newFile.close()

# The Rotation module with all learnable parameters
# Matrices from: https://en.wikipedia.org/wiki/Rotation_matrix
class Rotation(nn.Module):
	def __init__(self):
		super(Rotation, self).__init__()
		self.xr   = torch.rand(1) * 6.28 # Random x rotation
		self.yr   = torch.rand(1) * 6.28 # Random y rotation
		self.zr   = torch.rand(1) * 6.28 # Random z rotation
		self.bias = torch.randn(3)# Random movements in xyz directions
		self.xr.requires_grad_()  # This enables gradients for this parameter
		self.yr.requires_grad_()  # This enables gradients for this parameter
		self.zr.requires_grad_()  # This enables gradients for this parameter
		self.bias.requires_grad_()# This enables gradients for these parameters
	def forward(self, vecs):
		# Creates the x rotation matrix
		xmat = torch.zeros(3,3)
		xmat[0,0], xmat[1,1], xmat[2,2] = 1, self.xr.cos(), self.xr.cos()
		xmat[1,2], xmat[2,1]            =   -self.xr.sin(), self.xr.sin()
		# Creates the y rotation matrix
		ymat = torch.zeros(3,3)
		ymat[1,1], ymat[0,0], ymat[2,2] = 1, self.yr.cos(), self.yr.cos()
		ymat[2,0], ymat[0,2]            =   -self.yr.sin(), self.yr.sin()
		# Creates the z rotation matrix
		zmat = torch.zeros(3,3)
		zmat[2,2], zmat[0,0], zmat[1,1] = 1, self.zr.cos(), self.zr.cos()
		zmat[0,1], zmat[1,0]            =   -self.zr.sin(), self.zr.sin()
		# Apply the matrices
		vecs = torch.matmul(vecs, xmat)
		vecs = torch.matmul(vecs, ymat)
		vecs = torch.matmul(vecs, zmat)
		vecs = vecs + self.bias		
		return vecs
	# Returns all learnable parameters for optimizer
	def parameters(self):
		return [self.xr, self.yr, self.zr, self.bias]

pdb1_coordinates = extract(pdb1) # load xyz coordinates from 1st pdb
pdb2_coordinates = extract(pdb2) # load xyz coordinates from 2nd pdb

# Removes all missing amino-acids and make sure that the alined ammino-acids 
# have the same index. Specific for these two pdbs only...
pdb2_coordinates = [i[:-1]+[i[-1]-1] for i in pdb2_coordinates[1:-1] 
			 if (i[-1]-1) in [j[-1]for j in pdb1_coordinates]]

# Turns nested list object with xyz coordinates into torch tensor
pdb1_coordinates = torch.Tensor([i[:3]for i in pdb1_coordinates])
pdb2_coordinates = torch.Tensor([i[:3]for i in pdb2_coordinates])
pdb1_coordinates -= pdb1_coordinates.mean(0)
pdb2_coordinates -= pdb2_coordinates.mean(0)

rotation = Rotation() # initiate the rotation
optim = torch.optim.Adam(rotation.parameters()) # Create optimizer

# Train the parameters
for time in range(5000):
	optim.zero_grad() # Clear old gradients
	newCoordinates = rotation(pdb1_coordinates) 
	loss = F.mse_loss(newCoordinates, pdb2_coordinates) # What you try to optimize
	loss.backward() # Calculate gradients
	optim.step() # Update learnable parameters
	if time % 100==0: 
		print('Current loss:', float(loss))

# Save alligned pdb's coordinated for pymol
save('aligned', np.concatenate((newCoordinates.detach().numpy(),
								pdb2_coordinates.numpy())))
