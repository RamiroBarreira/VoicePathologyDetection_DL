# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:58:24 2022

@author: Ramiro R. A. Barreira
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean


loadModel = 0 # turn loadModel on & off
modelFileName = 'MEEI600'

NO = 2; #number of outputs (Neural Network)
EPOCHS = 600
PATIENCE = 50
weight_decay = 5e-4


print('reading files (training data)')
TRA = np.load('data/train_data.npz')
X = TRA['X'].copy()
Y = TRA['Y'].copy()
S = TRA['S'].copy()


print('reading files (validation data)')
TES = np.load('data/test_data.npz')
X_tes = TES['X_tes'].copy()
Y_tes = TES['Y_tes'].copy()
S_tes = TES['S_tes'].copy()



if loadModel == 1:
    model = tf.keras.models.load_model(modelFileName)
    model.summary()
elif loadModel == 0:
    model = tf.keras.models.Sequential()
    # =================== CAFFE NET =====================
    model.add(tf.keras.Input(shape=(227,227,3)))
    model.add(tf.keras.layers.Conv2D(96,(11,11),strides=(4,4),padding='valid',activation='relu',use_bias=True,kernel_regularizer=tf.keras.regularizers.l2(weight_decay),bias_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid'))
    model.add(tf.keras.layers.Conv2D(256,(5,5),padding='same',activation='relu',use_bias=True,kernel_regularizer=tf.keras.regularizers.l2(weight_decay),bias_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid'))
    model.add(tf.keras.layers.Conv2D(384,(3,3),padding='same',activation='relu',use_bias=True,kernel_regularizer=tf.keras.regularizers.l2(weight_decay),bias_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(tf.keras.layers.Conv2D(384,(3,3),padding='same',activation='relu',use_bias=True,kernel_regularizer=tf.keras.regularizers.l2(weight_decay),bias_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(tf.keras.layers.Conv2D(256,(3,3),padding='same',activation='relu',use_bias=True,kernel_regularizer=tf.keras.regularizers.l2(weight_decay),bias_regularizer=tf.keras.regularizers.l2(weight_decay)))
    #model.add(tf.keras.layers.Conv2D(256,(8,8),padding='valid',activation='relu',use_bias=True,kernel_regularizer=tf.keras.regularizers.l2(weight_decay),bias_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid')) 
    model.add(tf.keras.layers.Flatten())    
    model.add(tf.keras.layers.Dense(4096,activation='relu',use_bias=True,kernel_regularizer=tf.keras.regularizers.l2(weight_decay),bias_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4096,activation='relu',use_bias=True,kernel_regularizer=tf.keras.regularizers.l2(weight_decay),bias_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(tf.keras.layers.Dense(NO,activation='softmax',use_bias=True,kernel_regularizer=tf.keras.regularizers.l2(weight_decay),bias_regularizer=tf.keras.regularizers.l2(weight_decay)))
    
    model.summary()
    
    opt = tf.keras.optimizers.SGD(learning_rate=1e-3,momentum=0.9)
    model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE , restore_best_weights=True)
    history = model.fit(X,Y,batch_size=40,epochs=EPOCHS,shuffle=True,validation_data=(X_tes, Y_tes),callbacks=[callback])
    
    
        
    
    #PLOTS    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
        
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()



# val_loss, val_acc = model.evaluate(X,Y,verbose=0)
# print(val_loss)
# print(val_acc)

# val_loss, val_acc = model.evaluate(X_tes,Y_tes,verbose=0)
# print(val_loss)
# print(val_acc)








## computing training accuracy per segment
pred = model.predict(X,batch_size=None)
predict = np.argmax(pred,axis=1)
predict = np.array([predict])
predict = predict.T
matchArray = Y == predict
print('='*80)
print('train accuracy per segment: '+str(np.sum(matchArray)/matchArray.shape[0]))

## computing test accuracy per segment
pred_tes = model.predict(X_tes,batch_size=None)
predict_tes = np.argmax(pred_tes,axis=1)
predict_tes = np.array([predict_tes])
predict_tes = predict_tes.T
matchArray_tes = Y_tes == predict_tes
print('test accuracy per segment: '+str(np.sum(matchArray_tes)/matchArray_tes.shape[0]))





## computing training accuracy per file
P = np.array([]).reshape(0,pred.shape[1])
for i in range(S[len(S)-1,0]+1):
    if np.nonzero(S==i)[0].size != 0:
        P = np.vstack((P,gmean(pred[np.nonzero(S==i)[0]],axis=0)))
        #P = np.vstack((P,np.mean(pred[np.nonzero(S==i)[0]],axis=0)))

TARGET = np.array([]).reshape(0,1)
for i in range(S[len(S)-1,0]+1):
    if np.nonzero(S==i)[0].size != 0:
        TARGET = np.vstack((TARGET,Y[np.nonzero(S==i)[0][0]]))

pp = np.argmax(P,axis=1)
#TARGET = TARGET.T
matchArray = np.diag( pp == TARGET )
print('train accuracy per file: '+str(np.sum(matchArray)/matchArray.shape[0]))





## computing test accuracy per file
P_tes = np.array([]).reshape(0,pred_tes.shape[1])
P2_tes = np.array([]).reshape(0,1)
for i in range(S_tes[len(S_tes)-1,0]+1):
    if np.nonzero(S_tes==i)[0].size != 0:
        P_tes = np.vstack((P_tes,gmean(pred_tes[np.nonzero(S_tes==i)[0]],axis=0)))
        
        # computing and stacking P2_tes
        localClass0 = pred_tes[np.nonzero(S_tes==i)[0]]
        localClass = np.argmax(localClass0,axis=1) # provisorio
        numPAT = np.sum(localClass)
        numClass = localClass.size
        numNOR = numClass - numPAT
        if numNOR > numPAT:
            P2_tes = np.vstack((P2_tes,0))
        elif numNOR < numPAT:
            P2_tes = np.vstack((P2_tes,1))
        elif numNOR == numPAT:
            P2_tes = np.vstack((P2_tes,np.argmax(gmean(localClass0,axis=0))))



TARGET_tes = np.array([]).reshape(0,1)
for i in range(S_tes[len(S_tes)-1,0]+1):
    if np.nonzero(S_tes==i)[0].size != 0:
        TARGET_tes = np.vstack((TARGET_tes,Y_tes[np.nonzero(S_tes==i)[0][0]]))

pp_tes = np.argmax(P_tes,axis=1)
pp_tes = np.array([pp_tes])
pp_tes = pp_tes.T
pp2_tes = P2_tes
matchArray = pp_tes == TARGET_tes
matchArray2 = pp2_tes == TARGET_tes
print('test accuracy per file: '+str(np.sum(matchArray)/matchArray.shape[0]))
print('test accuracy per file (2nd method): '+str(np.sum(matchArray2)/matchArray2.shape[0]))
print('='*80)
