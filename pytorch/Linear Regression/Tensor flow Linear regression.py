import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
 Load the Boston housing dataset.
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
print(X_train.shape)
print("\n")
print("Input features: ", X_train[0])
print("\n")
print("Output target: ", y_train[0])
boston_features = {
    'Average Number of Rooms':5,
}
X_train_1d = X_train[:, boston_features['Average Number of Rooms']]
print(X_train_1d.shape)
X_test_1d = X_test[:, boston_features['Average Number of Rooms']]
plt.figure(figsize=(15, 5))
plt.xlabel('Average Number of Rooms')
plt.ylabel('Median Price [$K]')
plt.grid("on")
plt.scatter(X_train_1d[:], y_train, color='green', alpha=0.5);
model = Sequential()
# Define the model consisting of a single neuron.
model.add(Dense(units=1, input_shape=(1,)))
# Display a summary of the model architecture.
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=.005),
              loss='mse')
history = model.fit(X_train_1d, 
                    y_train, 
                    batch_size=16, 
                    epochs=101, 
                    validation_split=0.3)
def plot_loss(history):
    plt.figure(figsize=(20,5))
    plt.plot(history.history['loss'], 'g', label='Training Loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
    plt.xlim([0, 100])
    plt.ylim([0, 300])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # Predict the median price of a home with [3, 4, 5, 6, 7] rooms.
x = [3, 4, 5, 6, 7]
y_pred = model.predict(x)
for idx in range(len(x)):
    print("Predicted price of a home with {} rooms: ${}K".format(x[idx], int(y_pred[idx]*10)/10))
    # Generate feature data that spans the range of interest for the independent variable.
x = np.linspace(3, 9, 10)
# Use the model to predict the dependent variable.
y = model.predict(x)
def plot_data(x_data, y_data, x, y, title=None):
    plt.figure(figsize=(15,5))
    plt.scatter(x_data, y_data, label='Ground Truth', color='green', alpha=0.5)
    plt.plot(x, y, color='k', label='Model Predictions')
    plt.xlim([3,9])
    plt.ylim([0,60])
    plt.xlabel('Average Number of Rooms')
    plt.ylabel('Price [$K]')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plot_data(X_train_1d, y_train, x, y, title='Training Dataset')
    plot_data(X_test_1d, y_test, x, y, title='Test Dataset')
    