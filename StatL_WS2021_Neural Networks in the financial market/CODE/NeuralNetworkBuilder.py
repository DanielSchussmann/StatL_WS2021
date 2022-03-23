import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from DataPrep import *
#import sklearn
from sklearn.preprocessing import OneHotEncoder
data_training="market_data/AUD_USD.csv"
data_testing="market_data/AUD_CHF.csv"

def random_data_picker(data,minus,plus):
    picker = np.random.randint(0,len(data))
    hist_data=data[picker-minus:picker]
    imp_value=data[picker]
    epi_data=data[picker+1:picker+plus+1]

    return [hist_data,imp_value,epi_data]


def normalize(inp):
    A = [i*i for i in inp]
    v = (sum(A))**(0.5)
    A = [i / v for i in inp]
    return A
"""
def pick_and_relate(data,minus,plus):
    picker = np.random.randint(minus+1,len(data)-plus-1)
    imp_value = data[picker]
    hist_data = np.array(normalize(data[picker-minus:picker]/imp_value)).reshape(minus,)
    epi_data = np.array(normalize(data[picker+1:picker+plus+1]/imp_value)).reshape(plus,)
    return [imp_value, hist_data.tolist(), epi_data.tolist()]



def long_short_ident(data,minus,plus):
    picker = np.random.randint(minus+1,len(data)-plus-1)
    imp_value = data[picker]
    hist_data = np.array(normalize(data[picker-minus:picker]/imp_value)).reshape(minus,)
    bigger=np.array(np.where(imp_value<np.array(data[picker+1:picker+plus+1]))[0])
    smaller=np.array(np.where(imp_value>np.array(data[picker+1:picker+plus+1]))[0])
    #print(smaller.size/plus,bigger.size/plus)
   if bigger.size == smaller.size :
        epi_data=[0.0,0.1,0.0]#DO NOTHING

    else:
    epi_data=normalize([bigger.size/(plus),smaller.size/(plus)])
    #print(imp_value,np.array(data[picker + 1:picker + plus + 1]), np.array(np.where(imp_value < np.array(data[picker + 1:picker + plus + 1]))[0]))
    return [imp_value, hist_data.tolist(), epi_data]




data={'x':[],'y':[]}
for x in range(0,3200):
    pciked=long_short_ident(np.array(pd.read_csv('EURmajors/EURGBP_H.csv',usecols=[3])),10,6)
    data['x'].append(pciked[1])
    data['y'].append(pciked[2])
    pciked = long_short_ident(np.array(pd.read_csv('EURmajors/EURUSD_H.csv', usecols=[3])), 10, 6)
    data['x'].append(pciked[1])
    data['y'].append(pciked[2])
    pciked = long_short_ident(np.array(pd.read_csv('EURmajors/EURAUD_H.csv', usecols=[3])), 10, 6)
    data['x'].append(pciked[1])
    data['y'].append(pciked[2])


#data = [[np.array(pick_and_relate(np.array(pd.read_csv('EURmajors/EURUSD_H.csv',usecols=[3])),10,4)[1]), for x in range(0,100)]

print(data['y'])

with open('nn.pkl', 'rb') as f:
    data = pickle.load(f)
"""
#print('231ß38',data['x'][0])

class NeuralNetwork():
    def __init__(self):
        self.loss = 'binary_crossentropy'
        self.optimizer = 'adam'
        self.save_file = 'default'
        self.metrics = [tf.metrics.BinaryAccuracy(name='accuracy')]
        self.x_in = []
        self.y_in = []
        self.batch_size =25
        self.epochs = 50
        self.history =0
        self.model = tf.keras.Sequential()
        self.kernel_init = tf.keras.initializers.RandomNormal(stddev=0.01)
        self.bias_init = 'zeros'

    def add_layer(self,Type,Shape,Activation,Name,):
        if Type == 'Input':
            self.model.add(tf.keras.layers.Input(Shape, name=Name))
        elif Type == 'Dropout':
            self.model.add(tf.keras.layers.Dropout(Activation, input_shape=(Shape,), name=Name))
        elif Type == 'Hidden':
            self.model.add(tf.keras.layers.Dense(Shape, activation=Activation, name=Name, kernel_initializer = self.kernel_init, use_bias=True, bias_initializer=self.bias_init ))


        elif Type == 'Output':
            if Activation == None:
                self.model.add(tf.keras.layers.Dense(Shape, name=Name, kernel_initializer = self.kernel_init, use_bias=True, bias_initializer="zeros" ))
            else:
                self.model.add(tf.keras.layers.Dense(Shape, activation=Activation, name=Name , kernel_initializer = self.kernel_init, use_bias=True, bias_initializer="zeros" ))
        else:
            raise ValueError('Type {} is unknown to class'.format(Type))

        """
        elif Type == 'LSTM':
            self.model.add(tf.keras.layers.LSTM(Shape,activation=Activation,name=Name,use_bias=True,bias_initializer="zeros"))
        elif Type == 'GRU':
            self.model.add(tf.keras.layers.GRU(Shape,activation=Activation,name=Name))
        """
    def Compile(self,SHOW):
        self.model.build()
        self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = self.metrics)
        if SHOW == 'sum':
            self.model.summary()

        elif SHOW == 'model':
            tf.keras.utils.plot_model(self.model, to_file='tmp/'+self.save_file+'.png', show_shapes=True)
        elif SHOW == 'sum+model' or 'model+sum':
            #self.model.summary()
            tf.keras.utils.plot_model(self.model, to_file='tmp/' + self.save_file + '.png', show_shapes=True)


    def prep_data(self):
        self.x_train,self.x_val = np.split(np.array(self.x_in, dtype='float32'), 2 )
        self.y_train, self.y_val = np.split(np.array(self.y_in), 2)
        #print(self.x_val[0])
        #if len(self.x_in) != len(self.y_in) or self.x_train.size != self.x_val.size or self.y_train.size != self.y_val.size or self.x_train.size != self.y_train.size:
           # raise AttributeError('Oops training data does not really work')

    def Fit(self):
        fit= self.model.fit(
        self.x_train, self.y_train,
        batch_size = self.batch_size,
        epochs = self.epochs,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data = (self.x_val, self.y_val ))
        self.history = fit.history


    def Visualize(self):
        plt.plot(self.history['loss'],label='loss_function')
        plt.legend()
        plt.show()

    def Save(self):
        self.model.save('Neural_Networks/'+self.save_file)

    def Evaluate(self,data_x,data_y):
        print(self.history['accuracy'])
        #return self.model.evaluate(data_x, data_y)

    def Predict(self,data,start,end):
        return np.array(self.model.predict(data[start:end]))






#SMA Prediction

#optimization Network

def data_prep_10(input, am, backward, forward):
        picks = np.random.randint(0 + backward, len(input) - forward, size=am)
        data = {'x':[],'y':[]}


        [data['x'].append( normalize(np.array( input[picks[i] - backward: picks[i]] ).reshape((backward,)) )  )   for i in range(len(picks))]
        [data['y'].append(  ['LONG'] if np.mean(input[picks[i] + 1: picks[i] + forward + 1]) > input[picks[i]] else ['SHORT'] ) for i in range(len(picks))]
        enc= OneHotEncoder(sparse=False)
        data['y'] = enc.fit_transform(data['y'])
        return data


data = data_prep_10(np.array(pd.read_csv('EURmajors/EURGBP_H.csv', usecols=[4])), 8000, 30, 5)



test = NeuralNetwork()
#test.metrics=[tf.metrics.BinaryAccuracy(name='bin_accuracy')]
#test.loss = tf.keras.losses.CategoricalCrossentropy()
test.x_in=data['x']
test.y_in=data['y']
print(data['y'])

test.kernel_init= tf.keras.initializers.RandomNormal(stddev=0.01)
test.bias_init = tf.keras.initializers.RandomUniform(minval=0.01, maxval=0.03)
test.loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False,  name='crossentropy')
test.metrics=['accuracy']
test.add_layer('Dropout',30,0.2,'input')
test.add_layer('Hidden',32,'relu','Hidden_1')
test.add_layer('Hidden',32,'relu','Hidden_2')
#test.model.add(tf.keras.layers.GRU(20, input_shape=(10,),activation='relu'  ))
test.add_layer('Output',2,'sigmoid','out')
#print(test.model.weights)

#print(test.history['accuracy'])

test.Compile('sum')
test.prep_data()


test.Fit()
test.Visualize()


test_data =data_prep_10( np.array(pd.read_csv('EURmajors/EURGBP_H.csv', usecols=[4])), 32, 30, 5)

#print(np.array(test_data['x']).shape)
#results = test.model.predict_classes(np.array(test_data['x']))
y_prob = test.model.predict(test_data)

print(y_prob)







"""
plt.title('Loss')
test = NeuralNetwork()
test.metrics=[tf.metrics.BinaryAccuracy(name='accuracy')]
test.x_in=data['x']
test.y_in=data['y']
test.add_layer('Input',40,None,'input')
test.add_layer('Hidden',64,'sigmoid','Hidden_1')
test.add_layer('Hidden',64,'sigmoid','Hidden_2')
#test.model.add(tf.keras.layers.GRU(20, input_shape=(10,),activation='relu'  ))
test.add_layer('Output',4,None,'out')
print(test.model.weights)
print(test.model.layers[1].bias.numpy())
print(test.model.layers[1].bias_initializer)
test.Compile('sum')
test.prep_data()

test.Fit()
test.Visualize()



test_data={'x':[],'y':[]}
for x in range(0,100):
    pciked=next_candle(np.array(pd.read_csv('EURmajors/EURCHF_H.csv',usecols=[1,2,3,4])),10)
    test_data['x'].append(pciked[0])
    test_data['y'].append(pciked[1])

test.save_file='LSTM_1'
test.Save()
results = test.model.predict(test_data['x'])

print(results)
"""
