import numpy as np

class Game:
    def __init__(self):
        self.playerPos = 1
        
    def reset(self):
        self.playerPos = 1
        self.state = self.generateSample()
        return self.state
        
    def generateSample(self):
        self.obj = np.random.randint(0,3)
        self.x1 = 0 if self.playerPos == self.obj else 1
        self.x2 = np.random.randint(0,2)
        x = int(str(self.x1) + str(self.x2), 2)
        return x
    
    def randomChoice(self):
        return np.random.randint(0,2)
    
    def step(self, action):
        if action == 0:
            self.catch()
        elif action == 1:
            self.dodge()
            
        reward = 0.0
        done = False
        
        if self.playerPos == self.obj:
            if self.x2 == 0 and action == 0:
                reward += 1.0
            else:
                reward += 0.0
                done = True
        else:
            if self.x2 == 0:
                reward += 0.0
            elif self.x2 == 1 and action == 1:
                reward += 1.0
                
        new_state = self.generateSample()
        
        return new_state, reward, done
            
    def catch(self):
        self.playerPos = self.obj
        
    def dodge(self):
        if self.obj == self.playerPos:
            if self.playerPos == 0:
                self.playerPos = 1
            elif self.playerPos == 1:
                pos = np.random.randint(0,2)
                self.playerPos = pos * 2
            else:
                self.playerPos = 1