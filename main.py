import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback

# Custom callback to display live metrics after each epoch
class LiveMetrics(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: Loss = {logs['loss']:.4f}, Accuracy = {logs['accuracy']:.4f}")

def run():
    # Load the dataset
    df = pd.read_csv('data/sample.csv')
    X = df.drop('label', axis=1)
    y = df['label']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the features
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Convert labels to categorical (one-hot encoding)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(np.unique(y)))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(np.unique(y)))

    # Build a simple neural network model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(len(np.unique(y)), activation='softmax')  # Output layer with softmax for multi-class classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with live metrics
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, callbacks=[LiveMetrics()])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Final Test Accuracy: {accuracy:.2f}')

if __name__ == '__main__':
    run();