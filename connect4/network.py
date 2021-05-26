#%tensorflow_version 2.x
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Activation, Dropout, MaxPooling2D, Input, Add

class Network():
    def __init__(self):
        def generate_model():
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
            policyhead=Dense(7, activation='sigmoid')(policyhead)

            model = Model(inputs=input, outputs=[valuehead ,policyhead])
            return model
        self.model=generate_model()
        self.champion=generate_model()

    def new_champion(self):
        pass
    def still_champion(self):
        pass
    
    def predict(self, state, p):
        st=np.reshape(state, (1,state.shape[0],state.shape[1],-1) )
        inp1=(st*p>0).astype(np.float64)
        inp2=(st*p<0).astype(np.float64)
        inp=np.concatenate( (inp1,inp2 ), axis=3)
        return self.model.predict(inp)

    

if __name__ == "__main__":
    nn=Network()
    [v,p]=nn.predict(np.random.uniform(-1, 1, (6,7)), 1)
    print(v)
    print(p)
