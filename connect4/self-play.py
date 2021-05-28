import numpy as np
class selfplay():
    def __init__(self,game,nn,cput,num,batch):
        self.game=game
        self.nn=nn
        self.cput=cput
        self.num=num
        self.batch=batch
        #self.pnet=self.nnet.__class__(self.game)
        self.mcts=self.mcts(self.game,self.nn,self.cput,self.num)
        self.examples = []
        
    def play(self):
        examples = []
        while True:
            pi = self.mcts.getprobs()
            best=-float('inf')
            best_act=-1
            for a in range(len(pi)):
                if pi[a]>best:
                    best=pi[a]
                    best_act=a 
            examples.append([self.game.state,self.game.player,pi])
            r = self.game.end(self.game.state,self.game.player)
            if r!=-2:
                return [[x[0]*x[1] for x in examples], [r*self.game.player*x[1] for x in examples], [x[2] for x in examples]] 
            self.game.play(best_act)
    
    def learn(self):
        X, Y1, Y2 = []
        for _ in range(self.batch):
            x,y1,y2=self.play()
            X+=x
            Y1+=y1
            Y2+=y2
            self.game.state=np.zeros((6,7))
            self.game.player=1
        self.nn.train(X,Y1,Y2)
        self.nn.contender_vs_champion()
