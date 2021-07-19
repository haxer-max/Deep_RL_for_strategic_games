import numpy as np
import random
from copy import deepcopy

#from tensorflow.python.keras.backend_config import epsilon
class selfplay():
    def __init__(self):
        pass
        
    def play(self,toprint):
        examples = []
        self.game.state=np.zeros((6,7))
        self.game.player=1
        self.mcts=Mcts(self.game,self.nn,self.cput,self.num)
        while True:
            #self.mcts=Mcts(self.game,self.nn,self.cput,self.num)
            
            if toprint:
                print("""
                ##########################################
                ##########################################
                """)
                self.game.PrintBoard(self.game.state)
                print("")
            
            fstate = self.game.fstategen(self.game.state,self.game.player)
            pi = np.array(self.mcts.getprobs(deepcopy(fstate)))
            #print(pi)
            #probs=np.exp(pi)
            #print(probs)
            #probs=probs/np.sum(probs)
            #print(probs)
            best_act=np.random.choice(7,p=pi)

            self.game.play(best_act)
            r = self.game.end(self.game.state,self.game.player)
            if r!=-2:
                if toprint:
                    self.game.PrintBoard(self.game.state)
                    print("""
                    ---------------------------------------------------------------------
                    ---------------------------------------------------------------------
                    """)
                return [[x[0]*x[1] for x in examples], [r*self.game.player*x[1] for x in examples], [x[2] for x in examples]]

            examples.append([deepcopy(self.game.state),self.game.player,deepcopy(pi)])
    
    def learn(self,game,nn,cput,num,batch,toprint=False):
        self.game=game
        self.nn=nn
        self.cput=cput
        self.num=num
        self.batch=batch
        self.mcts=Mcts(self.game,self.nn,self.cput,self.num)
        X, Y1, Y2 = [],[],[]
        for _ in range(self.batch):
            print("batch",_)
            x,y1,y2=self.play(toprint)
            X+=x
            Y1+=y1
            Y2+=y2
            # for i in range(len(X)):
            #     print(X[i])
            #     print(Y1[i])
            #     print(Y2[i])
            
        self.nn.train(X,Y1,Y2,3)