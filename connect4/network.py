#%tensorflow_version 2.x
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Activation, Dropout, MaxPooling2D, Input, Add

class Network():
    def generate_model(self):
        def customLoss(y_true, y_pred):
            loss =tf.reduce_mean(tf.square(y_true[0]-y_pred[0])+ tf.reduce_sum(tf.math.multiply(y_true[1], tf.math.log(y_pred[1])), axis=1, keepdims=True) )
            return loss
        def identity_block(X, f, filters):
            F1, F2 = filters
            
            X_shortcut = X
            
            X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1))(X)
            X = Activation('relu')(X)
                
            X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same')(X)

            X = Add()([X_shortcut, X])
            X = Activation('relu')(X)
            return X
        input = Input(shape=(6,7,2))

        layer = Conv2D(filters = 8, kernel_size = (4, 4), strides = (1,1), activation='relu')(input)
        layer=identity_block(layer, 2, [8, 8])
        layer=identity_block(layer, 2, [8, 8])
        layer=Flatten()(layer)

        valuehead=layer
        policyhead=layer

        valuehead=Dense(32, activation='relu')(valuehead)
        valuehead=Dense(8, activation='relu')(valuehead)
        valuehead = Dropout(0.15)(valuehead)
        valuehead=Dense(4, activation='relu')(valuehead)
        valuehead=Dense(1, activation='tanh')(valuehead)

        policyhead=Dense(56, activation='relu')(policyhead)
        policyhead=Dense(42, activation='relu')(policyhead)
        policyhead=Dense(14, activation='relu')(policyhead)
        policyhead=Dense(7, activation='softmax')(policyhead)

        model = Model(inputs=input, outputs=[valuehead ,policyhead])
        opt = tf.keras.optimizers.SGD(momentum=0.1)
        model.compile(loss=self.customLoss, optimizer=opt)
        return model
    def __init__(self, path=None):
        if path==None:
            self.model=self.generate_model()
            self.champion=self.generate_model()

    def champ_predict(self, state, p):
        st=np.reshape(state, (1,state.shape[0],state.shape[1],-1) )
        inp1=(st*p>0).astype(np.float64)
        inp2=(st*p<0).astype(np.float64)
        inp=np.concatenate( (inp1,inp2 ), axis=3)
        return self.champion.predict(inp)

    def predict(self, state, p):
        st=np.reshape(state, (1,state.shape[0],state.shape[1],-1) )
        inp1=(st*p>0).astype(np.float64)
        inp2=(st*p<0).astype(np.float64)
        inp=np.concatenate( (inp1,inp2 ), axis=3)
        return self.model.predict(inp)
    
    def still_champ(self):
        self.champion.save("check.h5")
        tf.keras.backend.clear_session()        
        self.model = tf.keras.models.load_model("check.h5")
        self.champion = tf.keras.models.load_model("check.h5")

    def new_champ(self):
        self.model.save("check.h5")
        tf.keras.backend.clear_session()
        self.model = tf.keras.models.load_model("check.h5")
        self.champion = tf.keras.models.load_model("check.h5")
    
    # @tf.function
    # def train_step(self,x, y):
    #     with tf.GradientTape() as tape:
    #         logits = self.model(x, training=True)
    #         loss_value = loss_fn(y, logits)
    #     grads = tape.gradient(loss_value, model.trainable_weights)
    #     optimizer.apply_gradients(zip(grads, model.trainable_weights))
    #     train_acc_metric.update_state(y, logits)
    #     return loss_value
    
    # @tf.function
    # def test_step(self,x, y):
    #     val_logits = self.model(x, training=False)
    #     val_acc_metric.update_state(y, val_logits)

    @tf.function
    def customLoss(self,y_true, y_pred):
        loss =tf.reduce_mean(tf.square(y_true[0]-y_pred[0])+ tf.reduce_sum(tf.math.multiply(y_true[1], tf.math.log(y_pred[1])), axis=1, keepdims=True) )
        return loss

    def preprocess(self,X,Y1,Y2):
        X=np.array(X)
        Y1=np.array(Y1)
        Y2=np.array(Y2)
        X=np.reshape(X,(X[0],X[1],X[2],-1))
        inp1=(X>0).astype(np.float64)
        inp2=(X<0).astype(np.float64)
        X=np.concatenate( (inp1,inp2 ), axis=3)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X=X[indices]
        Y1 = Y1[indices]
        Y2 = Y2[indices]
        return X, [Y1,Y2]

    def train(self, data,Epochs=3, Batch=64):        
        X,Y=self.preprocess(data)
        self.model.fit(X, Y, batch_size=Batch, epochs=Epochs)
        pass
