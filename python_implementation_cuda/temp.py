import tensorflow as tf
from tensorflow import keras
import os
import time
tf.config.optimizer.set_jit(True)

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
start_time = time.time()

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"Found {len(physical_devices)} GPUs")
else:
    print("No GPU found, using CPU")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
print(x_train.shape)
# Convert labels to one-hot vectors
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


batch_size = 16
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),  # Flatten from (28,28,1) to (784,)
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="SGD",
    metrics=["accuracy"],
    jit_compile=True  # Enable XLA compilation
)
start_time = time.time()
model.fit(x_train,y_train, batch_size=batch_size,verbose=0,epochs=100)

# 6. Evaluate on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
# Print total execution time
print(f"Total execution time: {time.time() - start_time:.2f} seconds")