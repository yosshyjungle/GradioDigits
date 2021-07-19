import tensorflow as tf
import gradio as gr
import numpy as np

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)



def recognize_digit(img):
    img = img.reshape(1, 28, 28)
    prediction = model.predict(img).tolist()[0]
    return {str(i): prediction[i] for i in range(10)}

label = gr.outputs.Label(num_top_classes=4)
interface = gr.Interface(
    fn=recognize_digit,
    inputs='sketchpad',
    outputs=label,
#     live=True,
    title='Digit Recognizer'
)

interface.launch(share=True)


