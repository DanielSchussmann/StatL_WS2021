import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from DataPrep import *

data_training="market_data/AUD_USD.csv"
data_testing="market_data/AUD_CHF.csv"



x_example=mean_mvmnt(data_training,[4])[0]
y_example=mean_mvmnt(data_training,[4])[1]

x_train,x_val = np.split(np.array(x_example),2)
y_train,y_val = np.split(np.array(y_example),2)

#print(x_val[0:10],y_val[0:10])
x_test=np.array(mean_mvmnt(data_testing,[4])[0])
y_test=np.array(mean_mvmnt(data_testing,[4])[1])
#draw_test=mean_mvmnt(data_testing,[4])[2]

print(x_val[0])


inputs = tf.keras.Input(shape=(45,), name="digits")
x = tf.keras.layers.Dense(32, activation="relu", name="dense_1")(inputs)
x = tf.keras.layers.Dense(32, activation="relu", name="dense_2")(x)
outputs = tf.keras.layers.Dense(1, name="predictions")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.metrics.BinaryAccuracy(name='accuracy')])
#tf.keras.utils.plot_model(model, to_file='tmp/NN_1_visual.png', show_shapes=True)# Draw model
print(model.summary())

history=model.fit(
    x_train,y_train,
    batch_size=24,
    epochs=20,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val,y_val))
hist_dick=history.history

#model.save('Neural_Networks/relation_test')

accuracy = lambda x,y: model.evaluate(x_test[x:y],np.array(y_test[x:y]))
prediction=lambda x,y: np.array(model.predict(x_test[x:y]))



print(str(accuracy(0,100))+'%')
print(prediction(0,10))


plt.title('Loss')
plt.plot(hist_dick['loss'],color="coral")
plt.show()
