from collections import deque
import numpy as np

class CarPositions:
	
	def __init__(self):
		self.positions = []
		self.count = 0
		self.last_position = None
		self.average_position = None
		
	def update(self, new_position):
		## Check if difference in last and current position is less than 100, then consider it. Otherwise reset the average.
		if ((self.last_position is None) or (abs(new_position[2] - self.last_position[2])) < 100 and (abs(new_position[3] - self.last_position[3]) < 100)):
			self.average(new_position)
		else:
			self.reset()
			self.average(new_position)
		return self.average_position
		
	def average(self, new_position):
		if (self.count >= 10):
			self.positions.pop()

		self.count += 1
		self.positions.append(new_position)
		self.last_position = new_position
		self.average_position = np.mean(np.array(self.positions), axis = 0).astype(int)

	def reset(self):
		self.positions = []
		self.count = 0
		self.last_position = None
		self.average_position = None

	def get_average_position(self):
		return self.average_position