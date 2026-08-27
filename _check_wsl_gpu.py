import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("tf", tf.__version__)
print("gpus", gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    with tf.device("/GPU:0"):
        a = tf.random.normal((512, 512))
        b = tf.matmul(a, a)
        _ = b.numpy()
    print("gpu_matmul_ok")
else:
    print("NO_GPU")
