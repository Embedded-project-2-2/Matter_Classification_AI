import tensorflow as tf
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# 데이터 경로 설정
train_dir = './image_data/'

# 하이퍼파라미터 설정
image_size = (244, 244)  # 이미지 크기
batch_size = 32
epochs = 20  # 초기 에포크 수 (조기 종료로 조절)
learning_rate = 1e-4
validation_split = 0.2  # 전체 데이터의 20%를 검증 데이터로 사용

# 데이터 증강 설정 및 검증 데이터 분리
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=validation_split  # 검증 데이터로 사용할 비율 설정
)

# 학습 데이터 생성기
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # 훈련 데이터로 사용할 부분
)

# 검증 데이터 생성기
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # 검증 데이터로 사용할 부분
)

# EfficientNetB0 모델 로드 및 전이 학습용 설정
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(244, 244, 3))

# 기존 레이어를 고정
base_model.trainable = False

# 새로운 분류층 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)  # 중간 레이어
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # 출력 레이어

model = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 조기 종료 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 모델 학습
history = model.fit(
    train_generator,
    steps_per_epoch=math.ceil(train_generator.samples / batch_size),
    validation_data=val_generator,
    validation_steps=math.ceil(val_generator.samples / batch_size),
    epochs=epochs,
    callbacks=[early_stopping]  # 누락된 callbacks 추가
)

# 베이스 모델의 학습 레이어 일부를 푼 후 다시 학습 (옵션)
base_model.trainable = True
fine_tune_at = len(base_model.layers) // 2  # 절반 정도의 레이어부터 학습

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# 모델 재컴파일 (더 작은 학습률로)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 미세 조정(fine-tuning) 학습
history_fine = model.fit(
    train_generator,
    steps_per_epoch=math.ceil(train_generator.samples / batch_size),
    validation_data=val_generator,
    validation_steps=math.ceil(val_generator.samples / batch_size),
    epochs=epochs,  # 에포크 수 추가
    callbacks=[early_stopping]
)

# 최종 정확도 확인
final_loss, final_accuracy = model.evaluate(val_generator)
print(f"Final validation accuracy: {final_accuracy * 100:.2f}%")
