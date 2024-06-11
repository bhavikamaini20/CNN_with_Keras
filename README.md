# CNN_with_Keras
Here's a README file that you can use for your GitHub repository, describing your convolutional neural network (CNN) project for classifying the MNIST dataset:

---

# MNIST Digit Classification with Convolutional Neural Networks

This repository contains code for training and evaluating Convolutional Neural Networks (CNNs) on the MNIST dataset. The MNIST dataset consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9), each of size 28x28 pixels.

## Dependencies

The project requires the following Python packages:

- `tensorflow`
- `keras`

You can install these packages using pip:

```bash
pip install tensorflow keras
```

## Data Preparation

The MNIST dataset is automatically downloaded using the `keras.datasets` module. The images are then reshaped to include the channel dimension and normalized to have pixel values between 0 and 1.

```python
from keras.datasets import mnist

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255
X_test = X_test / 255
```

## Model Architecture

### Single Convolutional Layer

The model consists of:

- One convolutional layer with 16 filters of size 5x5 and ReLU activation
- One max pooling layer with pool size 2x2
- One fully connected layer with 100 neurons and ReLU activation
- One output layer with softmax activation

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

def convolutional_model_single():
    model = Sequential()
    model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train the model
model = convolutional_model_single()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
```

### Double Convolutional Layer

The model consists of:

- Two convolutional layers: the first with 16 filters of size 5x5 and the second with 8 filters of size 2x2, both with ReLU activation
- Two max pooling layers with pool size 2x2
- One fully connected layer with 100 neurons and ReLU activation
- One output layer with softmax activation

```python
def convolutional_model_double():
    model = Sequential()
    model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train the model
model = convolutional_model_double()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
```

## Results

The models are trained for 10 epochs each. The following are the results:

### Single Convolutional Layer

- **Accuracy**: 98.85%
- **Error**: 1.15%

### Double Convolutional Layer

- **Accuracy**: 98.74%
- **Error**: 1.26%

## Evaluation

The models are evaluated on the test dataset after training. Accuracy and error metrics are printed to show the performance of each model.

```python
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

Feel free to modify the content as per your requirements.
