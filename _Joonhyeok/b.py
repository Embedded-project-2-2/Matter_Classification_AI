import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from functools import partial
import matplotlib.pyplot as plt

# 이미지 크기 및 배치 크기 설정
img_height = 244
img_width = 244
batch_size = 32


# 데이터셋 로드
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "./image_data/",
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "./image_data/",
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
)

class_names = train_ds.class_names
print(class_names)


# 데이터 증식을 위한 레이어
data_augmentation = Sequential([
  layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),  # 좌우 반전
  layers.RandomRotation(0.25),    # 회전
  layers.RandomZoom(0.1),        # 확대/축소
  layers.RandomBrightness(0.1)   # 밝기 조절
])


# 정규화: 픽셀 값을 [0, 255] -> [0, 1] 범위로 변환
normalization_layer = layers.Rescaling(1./255)


# 학습 데이터에 전처리 및 증식 적용
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))


# 검증 데이터에도 정규화 적용 (증식은 학습 데이터에만 적용)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


# 데이터셋을 성능 향상을 위해 Prefetch 적용
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# EfficientNetB0 모델 불러오기 (사전 훈련된 가중치 사용)
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(img_height, img_width, 3)
)


# 모델의 일부 층은 고정
base_model.trainable = False


# 전체 모델 구성
model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])


base_model.trainable = True  # 미세 조정할 때 True로 설정
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# EarlyStopping과 LearningRateScheduler 정의
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-5 * 10 ** (epoch / 20)
)


# 모델 학습 (콜백 추가)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,  
    callbacks=[early_stopping, lr_scheduler]  # 콜백 추가
)


# 검증 정확도 확인
loss, accuracy = model.evaluate(val_ds)
print(f"Validation accuracy: {accuracy}")

model.summary()  # 모델의 구조


# 새로운 이미지 예측
img = image.load_img('./Classification_image/찌그러진캔.jpg', target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # 배치 차원 추가
predictions = model.predict(img_array)


# 예측된 클래스와 확률
predicted_class = np.argmax(predictions)
predicted_probability = predictions[0][predicted_class]

print(f"Predicted class: {class_names[predicted_class]}")
print(f"Probability: {predicted_probability * 100:.2f}%")


plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# 손실

plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

model.save('accuracy_85.keras')
