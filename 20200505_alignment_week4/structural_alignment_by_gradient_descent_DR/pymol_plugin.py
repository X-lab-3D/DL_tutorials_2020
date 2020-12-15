from pymol.cgo import COLOR, SPHERE, BEGIN, LINES, VERTEX, END
from pymol import cmd

"""
Pymol plugin
instalation: https://pymolwiki.org/index.php/Plugins
"""

def __init_plugin__(app):
	cmd.extend('showAlignment', showAlignment)

# This class is used to generate pdb backbones as connected spheres
class Mesh():
	""" Mesh class, each feature should be own Mesh object"""
	def __init__(self, name):
		self.name = name
		self.list = []
	def addSphere(self, vertex, size=0.1, color=[1.0,0.0,0.0]):
		self.list += [COLOR] + color 
		self.list += [SPHERE] + vertex + [size]
	def addLine(self, v1, v2, c1=[1,1,1]):
		self.list += [BEGIN, LINES]
		self.list += [COLOR] + c1
		self.list += [VERTEX] + v1
		self.list += [VERTEX] + v2
		self.list += [END]
	def push(self):
		"""When done defining mesh, send to pymol"""
		cmd.load_cgo(self.list, self.name, 1)

def showAlignment(fileName):
	radius = 0.5
	coordinates = open(fileName).read().split('\n')[:-1]
	coordinates = [[float(i)
					for i in line.split(' ')] 
					for line in coordinates]
	mesh = Mesh(fileName)
	prev = 0
	for i, coordinate in enumerate(coordinates):
		if i < len(coordinates)/2.:
			# Draws 1st protein
			mesh.addSphere(coordinate, size=radius, color=[1,0,0])
		else: 
			# Draws 2nd protein
			mesh.addSphere(coordinate, size=radius, color=[0,1,0])
		# Connect C-alphas backbone with lines
		if i<(len(coordinates)-1) and i != int(len(coordinates)/2.)-1:
			mesh.addLine(coordinates[i], coordinates[i+1])
	mesh.push()
