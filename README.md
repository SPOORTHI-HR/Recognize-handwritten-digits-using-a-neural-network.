# Recognize-handwritten-digits-using-a-neural-network.
Your code for recognizing handwritten digits using a neural network looks great! Here's a step-by-step explanation of what each part does:

# Step-by-Step Explanation:

1. import Libraries:
   ```python
   import tensorflow as tf
   import matplotlib.pyplot as plt
   ```

2. Load the MNIST Dataset:
   ```python
   mnist = tf.keras.datasets.mnist
   (x_train, y_train), (x_test, y_test) = mnist.load_data()
   ```
   - The MNIST dataset contains 60,000 training images and 10,000 testing images of handwritten digits (0-9).

3. Normalize Pixel Values:
   ```python
   x_train, x_test = x_train / 255.0, x_test / 255.0
   ```
   - Pixel values are normalized to be between 0 and 1 for better performance during training.

4. Define the Neural Network Model:
   ```python
   model = tf.keras.models.Sequential([
     tf.keras.layers.Flatten(input_shape=(28, 28)),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(10)
   ])
   ```
   - Flatten Layer: Converts the 28x28 pixel images into a 1D array of 784 pixels.
   - Dense Layer: A fully connected layer with 128 neurons and ReLU activation.
   - Dropout Layer: Prevents overfitting by randomly setting 20% of the input units to 0 at each update during training.
   - Output Layer: A dense layer with 10 neurons (one for each digit) without activation (logits).

5. Compile the Model:
   ```python
   loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
   model.compile(optimizer='adam',
                 loss=loss_fn,
                 metrics=['accuracy'])
   ```
   - Loss Function: Sparse categorical cross-entropy for multi-class classification.
   - Optimizer: Adam optimizer for efficient training.
   - Metrics: Accuracy to evaluate the model's performance.

6. Train the Model:
   ```python
   model.fit(x_train, y_train, epochs=5)
   ```
   - The model is trained for 5 epochs on the training data.

7. Evaluate the Model:
   ```python
   model.evaluate(x_test, y_test, verbose=2)
   ```
   - The model's performance is evaluated on the test data.

8. Make Predictions:
   ```python
   probability_model = tf.keras.Sequential([model, 
                                            tf.keras.layers.Softmax()])
   predictions = probability_model.predict(x_test)
   ```
   - A softmax layer is added to convert logits to probabilities.
   - Predictions are made on the test data.

9. Example Prediction:
   ```python
   print("Predicted digit:", tf.argmax(predictions[0]).numpy())
   plt.imshow(x_test[0], cmap='gray')
   plt.show()
   ```
   - The predicted digit for the first test image is printed.
   - The first test image is displayed using matplotlib.

This code should work well for recognizing handwritten digits.
