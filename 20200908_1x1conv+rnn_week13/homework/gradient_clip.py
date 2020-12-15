import torch

nmbr = 312 # the number you want to find the square root of.
start = torch.ones(1, requires_grad=True) # initial guess 

for epoch in range(50):
	loss = (start**2 - nmbr)**2 # the loss function
	loss.backward()
	#print('epoch %i:'%epoch,float(start.grad)) #-> printing gradients
	start.data -= start.grad * 0.01 # applying gradients with lr=0.01
	start.grad *= 0
	print('epoch %i:'%epoch, float(start))
	
	
	
