from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import ptre.tensorflow.keras as ptre

# PTRE: initialize PTRE.
ptre.init()
verbose = 1 if ptre.rank() == 0 else 0

(mnist_images, mnist_labels), _ = \
    tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % ptre.rank())

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
             tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.repeat().shuffle(10000).batch(128)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# PTRE: adjust learning rate based on number of GPUs.
opt = tf.optimizers.SGD(learning_rate=0.001, momentum=0.9)

# PTRE: add PTRE DistributedOptimizer.
opt = ptre.PullOptimizer(opt)

mnist_model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=opt,
                    metrics=['sparse_categorical_accuracy'],
                    experimental_run_tf_function=False
                    )

callbacks = [
    ptre.callbacks.BroadcastModelCallback(root_rank=0)
]

# Train the model.
# PTRE: adjust number of steps based on number of GPUs.
mnist_model.fit(dataset, steps_per_epoch=500, callbacks=callbacks, epochs=24, verbose=verbose)

ptre.finalize()
