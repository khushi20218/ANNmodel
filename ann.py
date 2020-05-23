import pandas as pd
dataset = pd.read_csv('Churn_Modelling.csv')
y = dataset['Exited']
X = dataset[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary']]
geo = dataset['Geography']
geo = pd.get_dummies(geo, drop_first=True )
gender = dataset['Gender']
gender = pd.get_dummies(gender, drop_first=True )
X = pd.concat([X,gender,geo], axis=1)
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
from keras.models import Sequential
model = Sequential()
from keras.layers import Dense
model.add(Dense(units=6, input_dim=11, activation='relu' ))
model.add(Dense(units=1,  activation='sigmoid' ))
model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy' , metrics=['accuracy'])
model.fit(X_train,y_train , epochs=30, validation_data=(X_test, y_test) )
accuracy = model.evaluate(X_test, y_test, verbose=0)
acc=accuracy[1]*100
print(acc)
model.save('ann.h5')





