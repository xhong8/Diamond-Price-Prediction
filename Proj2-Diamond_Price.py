# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:24:54 2022

@author: hong9
"""
#Import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import matplotlib.pyplot as plt
import datetime
import os

#1. Read CSV data
file_path = r"C:\Users\hong9\Documents\SHRDC\ML_learningMaterials\04-DeepLearning\Projects\Proj2-Diamond_Price\diamonds.csv"
diamond_data = pd.read_csv(file_path)

#2. Remove unwanted column
diamond_data = diamond_data.drop('Unnamed: 0', axis=1)

#3. Split the data into features and label
diamond_features = diamond_data.copy()
diamond_label = diamond_features.pop('price')

#4. Check the data
print("------------------Features-------------------------")
print(diamond_features.head())
print("-----------------Label----------------------")
print(diamond_label.head())

#%%
#5. Ordinal encode categorical features
cut_categories = ['Fair','Good','Very Good','Premium','Ideal']
color_categories = ['J','I','H','G','F','E','D']
clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
ordinal_encoder = OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])
diamond_features[['cut','color','clarity']] = ordinal_encoder.fit_transform(diamond_features[['cut','color','clarity']])
#Check the transformed features
print("---------------Transformed Features--------------------")
print(diamond_features.head())

#6. Split the data into train-validation-test sets, with a ratio of 60:20:20
SEED = 12345
x_train,x_iter,y_train,y_iter = train_test_split(diamond_features,diamond_label,test_size=0.4,random_state=SEED)
x_val,x_test,y_val,y_test = train_test_split(x_iter,y_iter,test_size=0.5,random_state=SEED)

#7. Perform feature scaling, using training data for fitting
standard_scaler = StandardScaler()
standard_scaler.fit(x_train)
x_train = standard_scaler.transform(x_train)
x_val = standard_scaler.transform(x_val)
x_test = standard_scaler.transform(x_test)

#Data preparation is completed at this step

#%%
#8. Create a feedforward neural network using TensorFlow Keras
number_input = x_train.shape[-1]
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=number_input))
model.add(tf.keras.layers.Dense(128,activation='elu'))
model.add(tf.keras.layers.Dense(64,activation='elu'))
model.add(tf.keras.layers.Dense(32,activation='elu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1))

#9.Compile the model
model.compile(optimizer='adam',loss='mse',metrics=['mae','mse'])

#%%
#10. Train and evaluate the model with validation data
#Define callback functions: EarlyStopping and Tensorboard
base_log_path = r"C:\Users\hong9\Documents\SHRDC\ML_learningMaterials\04-DeepLearning\Projects\Tensorboard\proj2_log"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=2)
EPOCHS = 100
BATCH_SIZE=64
history = model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks=[tb_callback,es_callback])


#%%
#11. Evaluate with test data for wild testing
test_result = model.evaluate(x_test,y_test,batch_size=BATCH_SIZE)
print(f"Test loss = {test_result[0]}")
print(f"Test MAE = {test_result[1]}")
print(f"Test MSE = {test_result[2]}")

#12. Plot a graph of prediction vs label on test data
predictions = np.squeeze(model.predict(x_test))
labels = np.squeeze(y_test)
plt.plot(predictions,labels,".")
plt.xlabel("Predictions")
plt.ylabel("Labels")
plt.title("Graph of Predictions vs Labels with Test Data")
save_path = r"C:\Users\hong9\Documents\SHRDC\ML_learningMaterials\04-DeepLearning\Projects\Proj2-Diamond_Price\images"
plt.savefig(os.path.join(save_path,"result.png"),bbox_inches='tight')
plt.show()