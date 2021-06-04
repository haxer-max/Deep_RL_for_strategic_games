import numpy as np
import random
from copy import deepcopy
from mcts import mcts
class selfplay():
    def __init__(self,game,nn,cput,num,batch, epsilon):
        self.epsilon=epsilon
        self.game=game
        self.nn=nn
        self.cput=cput
        self.num=num
        self.batch=batch
        #self.pnet=self.nnet.__class__(self.game)
        self.mcts=mcts(self.game,self.nn,self.cput,self.num)
        self.examples = []
        
    def play(self, toprint):
        examples = []
        while True:
            if toprint:
                self.game.PrintBoard(self.game.state)
                print("")
            pi = self.mcts.getprobs()
            best_act=-1
            if self.epsilon <random.uniform(0, 1):
                best=-float('inf')
                for a in range(len(pi)):
                    if pi[a]>best:
                        best=pi[a]
                        best_act=a 
            else:
                valid=self.game.ValidMoves(self.game.state, self.game.player)
                valid2=[]
                for i in range(6):
                    if valid[i]==1:
                        valid2.append(i)
                best_act=valid2[random.randint(0,len(valid2)-1)]

            examples.append([deepcopy(self.game.state),self.game.player,deepcopy(pi)])
            r = self.game.end(self.game.state,self.game.player)
            if r!=-2:
                if toprint:
                    self.game.PrintBoard(self.game.state)
                    print("""
                    ---------------------------------------------------------------------
                    ---------------------------------------------------------------------
                    """)
                return [[x[0]*x[1] for x in examples], [r*self.game.player*x[1] for x in examples], [x[2] for x in examples]] 
            self.game.play(best_act)
    
    def learn(self, toprint=False):
        X, Y1, Y2 = [],[],[]
        for _ in range(self.batch):
            x,y1,y2=self.play(toprint)
            x.pop()
            y1.pop()
            y2.pop()
            X+=x
            Y1+=y1
            Y2+=y2
            self.game.state=np.zeros((6,7))
            self.game.player=1
        #print(X)
        #print(Y1)
        #print(Y2)
        self.nn.train(X,Y1,Y2)
        #self.nn.contender_vs_champion()
