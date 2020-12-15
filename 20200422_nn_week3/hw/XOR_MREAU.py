import torch
from torchvision import datasets, transforms
from torch import optim
from torch import nn
import torch.nn.functional as F

class XOR(nn.Module):
	def __init__ (self):
		super (XOR, self).__init__()
		self.fc1 = nn.Linear(2,1000)
		self.fc2 = nn.Linear(1000,1)
	def forward(self, x):
		x = F.sigmoid(self.fc1(x))
		# can add nn.BtachNorm1d(x) to to normalize the output of the layer output
		x = F.sigmoid(self.fc2(x))
		return x           

def binary_acc(y_pred, y_test):
	y_pred_tag = torch.round(y_pred)
	correct_results_sum = (y_pred_tag == y_test).sum().float()
	acc = correct_results_sum/y_test.shape[0]
	acc = torch.round(acc * 100)
	return acc

def train(model, data, targets, lossfunction, optimizer, epochs=5, print_every=50):
	for e in range(epochs):
		epoch_loss = 0
		epoch_acc = 0
		# Pass training data through the model
		output = model.forward(data)
		# Compute the error and accuracy
		loss = lossfunction(output, targets)
		acc = binary_acc(output, targets)
		# Backward pass: compute gradient of the loss with respect to model parameters
		loss.backward()
		# Update the parameters and reinitialize the gradient
		optimizer.step()
		optimizer.zero_grad()
		# Save the loss info to check its evolution over epochs
		epoch_loss += loss.item()*data.size(0)
		epoch_acc += acc.item()
		# Print progression
		if e % print_every == 0:
			print(f'Epoch {e+0:03}: | Loss: {epoch_loss:.5f} | Acc: {epoch_acc:.3f}')	


# Load model 		
model = XOR()

# Define Loss function and optimizer
lossfunction = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create tensors (data and targets)
trainset = torch.Tensor([[0,1],[1,1],[1,0],[0,0]])
targets = torch.Tensor([1,0,1,0]).view(-1,1)

train(model, trainset, targets, lossfunction, optimizer, 200, 50)