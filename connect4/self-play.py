
class selfplay():
    def __init__(self,game,nnet,cput,num,batch):
        self.game=game
        self.nnet=nnet
        self.cput=cput
        self.num=num
        self.batch=batch
        #self.pnet=self.nnet.__class__(self.game)
        self.mcts=mcts(self.game,self.nnet,self.cput,self.num)
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
            examples.append([self.game.state,self.game.player,pi,None])
            r = self.game.end(self.game.state,self.game.player)
            if r!=-2:
                return [(x[0],x[2],(r if x[1]==self.game.player else (-r))) for x in examples]
            self.game.play(best_act)
    
    def learn(self):
        trainingexamples = []
        for _ in range(self.batch):
            trainingexamples+=self.play()
            game.state=np.zeros((6,7))
            game.player=1
        nn.train(trainingexamples)
        nn.contender_vs_champion()
