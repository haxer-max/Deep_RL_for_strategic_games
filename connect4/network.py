#%tensorflow_version 2.x
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Activation, Dropout, MaxPooling2D, Input, Add, BatchNormalization, LeakyReLU

class Network():
    def generate_model(self):
        def identity_block(X, f, filters):
            F1, F2 = filters
            
            X_shortcut = X
            
            X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1))(X)
            X = LeakyReLU()(X)
                
            X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same')(X)
            X = BatchNormalization()(X)
            X = Add()([X_shortcut, X])
            X = LeakyReLU()(X)
            return X
        input = Input(shape=(6,7,2))

        layer = Conv2D(filters = 8, kernel_size = (1, 1), strides = (1,1))(input)
        layer=LeakyReLU()(layer)
        layer=identity_block(layer, 4, [8, 8])
        layer=identity_block(layer, 3, [8, 8])
        layer=MaxPooling2D(pool_size=(3,3),strides=(1,1))(layer)

        layer = Conv2D(filters = 16, kernel_size = (1, 1), strides = (1,1))(layer)
        layer=LeakyReLU()(layer)
        layer=identity_block(layer, 3, [16, 16])
        layer=identity_block(layer, 2, [16, 16])
        layer=MaxPooling2D(pool_size=(2,2),strides=(1,1))(layer)
        layer = Conv2D(filters = 32, kernel_size = (3, 3), strides = (1,1))(layer)
        layer=LeakyReLU()(layer)

        layer=Flatten()(layer)

        #--------VALUE NETWORK---------
        valuehead=Dense(32)(layer)
        layer=LeakyReLU()(layer)
        valuehead = Dropout(0.4)(valuehead)

        valuehead=Dense(8)(valuehead)
        valuehead=LeakyReLU()(valuehead)
        valuehead = Dropout(0.2)(valuehead)
        
        valuehead=Dense(4)(valuehead)
        valuehead=LeakyReLU()(valuehead)

        valuehead=Dense(1, activation='tanh', name="valuehead")(valuehead)


        #--------POLICY NETWORK---------
        policyhead=Dense(56)(layer)
        policyhead=LeakyReLU()(policyhead)
        policyhead = Dropout(0.3)(policyhead)

        policyhead=Dense(42)(policyhead)
        policyhead=LeakyReLU()(policyhead)
        policyhead = Dropout(0.1)(policyhead)

        policyhead=Dense(14)(policyhead)
        policyhead=LeakyReLU()(policyhead)

        policyhead=Dense(7, activation='softmax',name="policyhead")(policyhead)

        model = Model(inputs=input, outputs=[valuehead ,policyhead])
        opt = tf.keras.optimizers.SGD(momentum=0.1)
        loss1 = tf.keras.losses.MeanSquaredError()
        loss2 = self.customLoss
        losses = {
            "valuehead": loss1,
            "policyhead": loss2,
        }
        model.compile(loss=losses, optimizer=opt)
        return model

    def __init__(self, path=None):
        if path==None:
            self.model=self.generate_model()
            self.champion=self.generate_model()
        else:
            self.model = tf.keras.models.load_model(path)
            self.champion = tf.keras.models.load_model(path)

    def champ_predict(self, state, p):
        st=np.reshape(state, (1,state.shape[0],state.shape[1],-1) )
        inp1=(st*p>0).astype(np.float32)
        inp2=(st*p<0).astype(np.float32)
        inp=np.concatenate( (inp1,inp2 ), axis=3)
        return self.champion.predict(inp)

    def predict(self, state, p):
        st=np.reshape(state, (1,state.shape[0],state.shape[1],-1) )
        inp1=(st*p>0).astype(np.float32)
        inp2=(st*p<0).astype(np.float32)
        inp=np.concatenate( (inp1,inp2 ), axis=3)
        #print(inp[0,:,:,0])
        #print(inp[0,:,:,1])
        return self.model.predict(inp)
    
    def still_champ(self,name="check.h5"):
        self.champion.save(name)
        tf.keras.backend.clear_session()        
        self.model = tf.keras.models.load_model(name)
        self.champion = tf.keras.models.load_model(name)

    def new_champ(self,name="check.h5"):
        self.model.save(name)
        tf.keras.backend.clear_session()        
        self.model = tf.keras.models.load_model(name)
        self.champion = tf.keras.models.load_model(name)
    
    @tf.function
    def customLoss(self,y_true, y_pred):
        #print(y_true)
        #print(y_pred)
        loss =tf.reduce_mean(tf.reduce_sum(-tf.math.multiply(y_true, tf.math.log(y_pred)), axis=1, keepdims=True) )
        return loss

    def preprocess(self,X,Y1,Y2):
        X=np.array(X)
        Y1=np.array(Y1).astype(np.float32)
        Y2=np.array(Y2).astype(np.float32)
        X=np.reshape(X,(X.shape[0],X.shape[1],X.shape[2],-1))
        inp1=(X>0).astype(np.float32)
        inp2=(X<0).astype(np.float32)
        X=np.concatenate( (inp1,inp2 ), axis=3)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X=X[indices]
        Y1 = Y1[indices]
        Y2 = Y2[indices]
        return X,Y1,Y2

    def train(self, X,Y1,Y2,Epochs=3, Batch=64):        
        X,Y1,Y2=self.preprocess(X, Y1, Y2)
        Y= {"valuehead": Y1,
            "policyhead": Y2 }
        #print(X.shape)
        #print(X[5,:,:,0])
        #print(X[5,:,:,1])
        #print(Y)
        self.model.fit(X, Y, batch_size=Batch, epochs=Epochs)
        pass