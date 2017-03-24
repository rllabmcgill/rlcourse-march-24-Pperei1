import numpy as np
import random
import matplotlib.pyplot as plt

class RandomWalk:
	def __init__(self,size):
		self.size = size-1;
		self.pos = int((size-1)/2)
	
	def playMove(self):
		self.pos = self.pos + random.randint(-50,50)
			
	def playGame(self,weights,alpha):
		while(self.pos != 0 and self.pos != self.size):
			[previousPos,nextPos,reward] = self.getUpdate()
			weights = self.updateWeightsSGTD(weights,reward,previousPos,nextPos,alpha)
		return weights
		
	def playGameLTD(self,weights,w,alpha,beta):
		while(self.pos != 0 and self.pos != self.size):
			[previousPos,nextPos,reward] = self.getUpdate()
			[weights,w] = self.updateWeightsLTD(weights,w,reward,previousPos,nextPos,alpha,beta)
		return [weights,w]
	
	def updateWeightsLTD(self,weights,w,reward,previousPos,nextPos,alpha,beta):
			[previous,next] = self.makeFeatureVectors(previousPos,nextPos)
			delta = reward + np.dot(weights,next) - np.dot(weights,previous)
			weights = weights + alpha*delta*previous - alpha*next*(np.dot(previous,w))
			w = w + beta*(delta - np.dot(previous,w))*previous
			return [weights,w]
			
	def updateWeightsSGTD(self,weights,reward,previousPos,nextPos,alpha):
			[previous,next] = self.makeFeatureVectors(previousPos,nextPos)
			#print(alpha*(reward + np.dot(weights,next) - np.dot(weights,previous))*previous)
			weights = weights + alpha*(reward + np.dot(weights,next) - np.dot(weights,previous))*previous
			return weights
			
	
	def makeFeatureVectors(self,previousPos,nextPos):
			previous = np.array([0.0]*10)
			previous[previousPos] = 1.0
			next = np.array([0.0]*10)
			next[nextPos] = 1.0
			return [previous,next]
			
	def getUpdate(self):
		previousPos = int(self.pos/100)
		self.playMove()
		if self.pos <= 0 :
			reward = -1
			self.pos = 0
		elif self.pos >= self.size:
			reward = 1
			self.pos = self.size
		else:
			reward = 0
		nextPos = int(self.pos/100)
		return [previousPos,nextPos,reward]
	
class RandomPlayer:
	def __init__(self,weights,w):
		self.weights = weights
		self.w = w
		
	def training(self,alpha,beta,numberOfGames,numberOfStates):
		for i in range(0,numberOfGames):
			rw = RandomWalk(numberOfStates)
			self.weights = rw.playGame(self.weights,alpha)
			

weights = np.array([0.0]*10)
w = np.array([0.0]*10)

alphaV = [0.000001,0.00003,0.001,0.003,0.01,0.03,0.1]
for i in range(0,7):
	print(i)
	rp = RandomPlayer(weights,w)
	rp.training(alphaV[i],alphaV[i],1000,1000)		
	plt.plot(rp.weights,label = "alpha: "+str(alphaV[i]))
	
plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)
plt.show()


alphaV = [0.000001,0.00003,0.001,0.003,0.01,0.03,0.1]
for i in range(0,7):
	print(i)
	rp = RandomPlayer(weights,w)
	rp.training(alphaV[i],alphaV[i],1000,1000)		
	plt.plot(rp.weights,label = "alpha: "+str(alphaV[i]))
	
plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)
plt.show()
