import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import math


def OneD_data(inputfile,columns):
    data = pd.read_csv(inputfile, usecols=columns)
    df = data.copy()

    df=np.array(np.array_split(df,int(len(df)/10))).reshape([int(len(df)/10),10])#testing examples

    x_raw =df.copy()
    x_norm=df[::2]
    #for i in range(0,len(x_norm)):
     #   x_norm[i]=tf.keras.utils.normalize(x_norm[i]) #make it so the actual value of the currency doesnt matter.

    y_norm=[]
    for i in range(0,len(df[1::2])):
        if max(df[1::2][i]) - df[1::2][i][0] > df[1::2][i][0] - min(df[1::2][i]): #if the max is bigger than the // change is the same still needs implementation
            y_norm.append(1)
        else:
            y_norm.append(-1)

    return [x_norm,y_norm,x_raw]



"""def mean_movement(inputfile,columns):
    data = pd.read_csv(inputfile, usecols=columns)
    df = data.copy()

    df = np.array(np.array_split(df, int(len(df) / 10))).reshape([int(len(df) / 10), 10])  # testing examples

    x_raw = df.copy()
    x_norm = df[::2]
    for i in range(0,len(x_norm)):
       x_norm[i]=tf.keras.utils.normalize(x_norm[i]) #make it so the actual value of the currency doesnt matter.

    y_norm = []
    for i in range(0, len(df[1::2])):
        y_norm.append(np.mean(df[1::2][i]))

    return [x_norm, y_norm, x_raw]"""

def normalize(inp):
    A = [i*i for i in inp]
    v = (sum(A))**(0.5)
    A = [i / v for i in inp]
    return A

def relationfy(x):
    output = []
    no_order=[]
    #viz=[]
    for j in range(len(x)-1):
        temp=[]
        #color=(np.random.random(), np.random.random(), np.random.random())
        for i in range(j+1,len(x)):
            temp.append(x[j]/x[i])
            no_order.append(x[j]/x[i])
            #viz.append([[j,i],[x[j],x[i]]])
        output.append(temp)
    return [output,normalize(no_order)]


def mean_mvmnt(inputfile,columns):
    data = pd.read_csv(inputfile, usecols=columns)
    df = data.copy()

    df = np.array(np.array_split(df, int(len(df) / 10))).reshape([int(len(df) / 10), 10])  # testing examples

    x_norm = []
    for i in range(0, len(df[::2])):
        x_norm.append(relationfy(df[::2][i])[1])

    y_norm = []
    for i in range(0, len(df[1::2])):
        y_norm.append(np.mean(normalize(df[1::2][i])))

    return [x_norm, y_norm, df]

#print(mean_mvmnt('market_data/AUD_CHF.csv',[4])[0][0:10])









def mean_movement(inputfile,columns):
    data = pd.read_csv(inputfile, usecols=columns)
    df = data.copy()

    df = np.array(np.array_split(df, int(len(df) / 10))).reshape([int(len(df) / 10), 10])  # testing examples
    x_raw = df.copy()
    for i in range(0, len(df)):
        df[i]=normalize(df[i])

    x_norm = df[::2]
    y_norm = []
    for i in range(0, len(df[1::2])):
        y_norm.append(np.mean(df[1::2][i]))

    return [x_norm, y_norm, df,x_raw]





def data_prep_10(input,am,backward,forward,):
        picks = np.random.randint(0+backward,len(input)-forward,size=am)
        data = [ [input[picks[i]-backward : picks[i]], 1 if np.mean(input[picks[i]+1 : picks[i]+forward+1])> input[picks[i]] else -1 ] for i in range(len(picks)) ]


        return data


print(data_prep_10( np.array(pd.read_csv('EURmajors/EURGBP_H.csv',usecols=[4])),100,10,5))









#print(mean_movement('market_data/AUD_CHF.csv',[4]))
"""
import pickle

def next_candle(data,minus):
    picker = np.random.randint(minus + 1, len(data)-1)
    imp_value = data[picker][3]
    hist_data = np.array(normalize(data[picker - minus:picker] / imp_value)).reshape(minus*4, )
    deNormFaktor=10
    epi_data = np.array(normalize(data[picker+1] / imp_value))
    return [hist_data.tolist(), epi_data.tolist(),deNormFaktor]

ting=next_candle(np.array(pd.read_csv('EURmajors/EURGBP_H.csv',usecols=[1,2,3,4])),10)

print(ting)
data={'x':[],'y':[]}
for x in range(0,1000):
    pciked=next_candle(np.array(pd.read_csv('EURmajors/EURGBP_H.csv',usecols=[1,2,3,4])),10)
    data['x'].append(pciked[0])
    data['y'].append(pciked[1])
    pciked = next_candle(np.array(pd.read_csv('EURmajors/EURUSD_H.csv', usecols=[1,2,3,4])), 10)
    data['x'].append(pciked[0])
    data['y'].append(pciked[1])
    pciked = next_candle(np.array(pd.read_csv('EURmajors/EURAUD_H.csv', usecols=[1,2,3,4])), 10)
    data['x'].append(pciked[0])
    data['y'].append(pciked[1])

with open('nn.pkl', 'wb') as f:
    pickle.dump(data, f)"""









#df.to_csv('market_data/EUR_USD_D_corrected.csv')
def long_short_ident(data,minus,plus):
    picker = np.random.randint(minus+1,len(data)-plus-1)
    imp_value = data[picker]
    hist_data = np.array(normalize(data[picker-minus:picker]/imp_value)).reshape(minus,)
    bigger=np.array(np.where(imp_value<np.array(data[picker+1:picker+plus+1]))[0])
    smaller=np.array(np.where(imp_value>np.array(data[picker+1:picker+plus+1]))[0])
    #print(smaller.size/plus,bigger.size/plus)
    if bigger.size < smaller.size :
        epi_data=[1.0,0.0]#DO NOTHING

    elif bigger.size > smaller.size:
        epi_data = [0.0, 1.0]
    else:
        epi_data=[0.0,0.0]
    #epi_data=[bigger.size/plus,smaller.size/plus]

    #print(imp_value,np.array(data[picker + 1:picker + plus + 1]), np.array(np.where(imp_value < np.array(data[picker + 1:picker + plus + 1]))[0]))
    return [imp_value, hist_data.tolist(), epi_data]

"""
data={'x':[],'y':[]}
for x in range(0,500):
    pciked=long_short_ident(np.array(pd.read_csv('EURmajors/EURGBP_H.csv',usecols=[3])),10,6)
    data['x'].append(pciked[1])
    data['y'].append(pciked[2])
    pciked = long_short_ident(np.array(pd.read_csv('EURmajors/EURUSD_H.csv', usecols=[3])), 10, 6)
    data['x'].append(pciked[1])
    data['y'].append(pciked[2])
    pciked = long_short_ident(np.array(pd.read_csv('EURmajors/EURAUD_H.csv', usecols=[3])), 10, 6)
    data['x'].append(pciked[1])
    data['y'].append(pciked[2])

with open('nn.pkl', 'wb') as f:
    pickle.dump(data, f)


"""








def angular_price_movement(inputfile,columns):
    data = pd.read_csv(inputfile, usecols=columns)
    df = data.copy()

    df=np.array(np.array_split(df,int(len(df)/10))).reshape([int(len(df)/10),10])#testing examples

    x_raw =df.copy()
    x_norm=df[::2]
    #for i in range(0,len(x_norm)):
     #   x_norm[i]=tf.keras.utils.normalize(x_norm[i]) #make it so the actual value of the currency doesnt matter.

    y_norm=[]
    for i in range(0, len(df[1::2]) ):
        dom_change=max(max(df[1::2][i])-df[1::2][i][0] , df[1::2][i][0]-min(df[1::2][i]))
        y_norm.append(math.asin( dom_change / abs(df[1::2][i][0] - dom_change) ) )

    return [x_norm,y_norm,x_raw]

