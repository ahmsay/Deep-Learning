import numpy as np
from NeuralNetwork import NeuralNetwork

class NeuralRacer:
    def __init__(self):
        self.playerPos = 0.5
        self.setPos(self.playerPos)
        self.x = np.array([], dtype=np.int32).reshape(0,2)
        self.y = np.array([], dtype=np.int32).reshape(0,1)
        self.score = 0
        self.nn = NeuralNetwork(2,3,1)
    
    def setPos(self, index):
        self.playerPos = index
        index = np.int32(index * 2)
        
    def move(self, index):
        if index == 1:
            if self.playerPos == 0:
                self.setPos(0.5)
            elif self.playerPos == 0.5:
                pos = np.random.randint(0,2)
                self.setPos(pos)
            else:
                self.setPos(0.5)
    
    def run(self):
        for i in range(25):
            print("Generation", i)
            for i in range(100):
                obstacle = np.random.randint(0,3) / 2
                value = [self.playerPos,obstacle]
                decision = 1 if self.playerPos == obstacle else 0
                result_x = np.array([value])
                pred = np.random.uniform(0, 1)
                if value in self.x.tolist():
                    pred = self.nn.predict(result_x)
                pred = 0 if pred < 0.5 else 1
                result_y = np.array([[decision]])
                self.move(pred)
                if pred == decision:
                    self.score += 1
                    if not value in self.x.tolist():
                        self.x = np.concatenate((self.x, result_x), axis = 0)
                        self.y = np.concatenate((self.y, result_y), axis = 0)
                else:
                    self.nn.setWeights()
                    self.nn.update(self.x, self.y, 1000, 1)
                    break
            print("Score: ", self.score)
            self.score = 0
                
nr = NeuralRacer()
nr.run()
x = nr.x
y = nr.y