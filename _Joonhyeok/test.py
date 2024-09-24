import tensorflow as tf

# TensorFlow에서 사용 가능한 GPU 확인
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"사용 가능한 GPU: {gpus}")
else:
    print("GPU가 감지되지 않았습니다.")
