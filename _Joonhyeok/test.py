import tensorflow as tf

# 사용 가능한 GPU 확인
print("GPU 사용 가능 여부: ", tf.config.list_physical_devices('GPU'))


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.set_logical_device_configuration(
        gpu,
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
    )
