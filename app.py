import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential

data = pd.read_csv('data/churn.csv')
# get details about the dataset
print(data.info())

X = data.iloc[:,3:13]
y = data.iloc[:,13]

print(f"Shape of X is {X.shape} shape of y is {y.shape}")

geography = pd.get_dummies(X['Geography'],drop_first=True)
gender = pd.get_dummies(X['Gender'],drop_first=True)

X = pd.concat([X,geography,gender],axis=1)

X = X.drop(['Geography','Gender'],axis=1)

# spilt train and test data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# apply feature scaling

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# create the model 
model = Sequential()

# create the hidden layers
hl1 = Dense(units=6,activation='relu',kernel_initializer='he_uniform',input_dim = 11)
hl2 = Dense(units=6,activation='relu',kernel_initializer='he_uniform')

# create the output layer
output_layer = Dense(units=1,activation='sigmoid',kernel_initializer='glorot_uniform')

model.add(hl1)
model.add(hl2)
model.add(output_layer)

# compile the neural network

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# fit the ann to the training data

model.fit(X_train,y_train,batch_size=16,epochs=100,validation_split=0.33)

# predict the output from the model 

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# get the accuracy score of the model 

score = accuracy_score(y_test,y_pred)
print(f"Accuracy of the A.N.N is {score}")

