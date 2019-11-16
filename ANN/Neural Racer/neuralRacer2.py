import numpy as np
from NeuralNetwork import NeuralNetwork

class NeuralRacer:
    def __init__(self):
        self.playerPos = 0.5
        self.x = np.array([], dtype=np.int32).reshape(0,2)
        self.y = np.array([], dtype=np.int32).reshape(0,1)
        self.score = 0
        self.nn = NeuralNetwork(2,4,4,1)
    
    def catch(self):
        self.playerPos = self.obj
        
    def dodge(self):
        if self.playerPos == 0:
            self.playerPos = 0.5
        elif self.playerPos == 0.5:
            pos = np.random.randint(0,2)
            self.playerPos = pos
        else:
            self.playerPos = 0.5
    
    def run(self):
        for i in range(25):
            print("Generation", i+1)
            for i in range(100):
                self.obj = np.random.randint(0,3) / 2
                x1 = 0 if self.playerPos == self.obj else 1
                x2 = np.random.randint(0,2)
                value = [x1, x2]
                result_x = np.array([value])
                pred = np.random.randint(0,3) / 2
                if value in self.x.tolist():
                    pred = self.nn.predict(result_x)
                    if pred < 0.25:
                        pred = 0
                    elif pred >= 0.25 and pred < 0.75:
                        pred = 0.5
                    else:
                        pred = 1
                    
                result_y = np.array([[pred]])
                if pred == 0.5:
                    self.catch()
                elif pred == 1:
                    self.dodge()
                score = 0
                if (x2 == 0 and pred == 0.5): score += 1
                if (x1 == 0 and x2 == 1 and pred == 1): score += 1
                if (x1 == 1 and x2 == 1 and pred == 0): score += 1
                if score == 1:
                    self.score += 1
                    if not value in self.x.tolist():
                        self.x = np.concatenate((self.x, result_x), axis = 0)
                        self.y = np.concatenate((self.y, result_y), axis = 0)
                else:
                    self.nn.setWeights()
                    self.nn.update(self.x, self.y, 5000, 1)
                    break
            print("Score:", self.score)
            self.score = 0

nr = NeuralRacer()
nr.run()
x = nr.x
y = nr.y