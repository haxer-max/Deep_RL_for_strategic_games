import numpy as np
import math
from copy import deepcopy

class Mcts():
    def __init__(self,game,nn,cput,num):
        self.game = game
        self.nn = nn
        self.cput = cput
        self.num = num
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
    
    def search(self,fstate):
        s = fstate.tobytes()
        
        if s not in self.Es:
            self.Es[s] = self.game.end(fstate,1)
        if self.Es[s]!=-2:
            return -self.Es[s]
        
        if s not in self.Ps:
            [v,tempP] = self.nn.predict(fstate,1)
            v = v[0][0]
            self.Ps[s] = tempP[0]
            valids = self.game.ValidMoves(fstate,1)
            self.Ps[s] = self.Ps[s]*valids
            sPs = np.sum(self.Ps[s])
            if sPs > 0:
                self.Ps[s] = self.Ps[s]/sPs
            else:
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] = self.Ps[s]/np.sum(self.Ps[s])
            
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v
        
        valids = self.Vs[s]
        best = -float('inf')
        best_act = -1
        for a in range(7):
            if valids[a]:
                if (s,a) in self.Qsa:
                    PUCT = self.Qsa[(s,a)] + self.cput * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1+self.Nsa[(s,a)])
                else:
                    PUCT = self.cput * self.Ps[s][a] * math.sqrt(self.Ns[s])
                
                if PUCT > best:
                    best = PUCT
                    best_act = a
        
        a = best_act
        next_state,next_player = self.game.Next(fstate,1,best_act)
        next_fstate = self.game.fstategen(next_state,next_player)
        
        v = self.search(deepcopy(next_fstate))
        
        if (s, a) in self.Qsa:
            self.Qsa[(s,a)]=(self.Nsa[(s,a)]*self.Qsa[(s,a)]+v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)]+=1

        else:
            self.Qsa[(s,a)]=v
            self.Nsa[(s,a)]=1

        self.Ns[s]+=1
        return -v
    
    def getprobs(self,fstate):
        
        for i in range(self.num):
            self.search(deepcopy(fstate))
        
        s = fstate.tobytes()
        probs = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(7)]
        
        probSum = float(sum(probs))
        if probSum==0:
            return probs
        else:
            probs = [x/probSum for x in probs]
        return probs
    
    def getAction(self,fstate):
        probs=self.getprobs(fstate)
        best=-float('inf')
        best_act=-1
        for a in range(len(probs)):
            if probs[a]>best:
                best=probs[a]
                best_act=a
        return best_act