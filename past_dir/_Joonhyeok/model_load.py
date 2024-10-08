import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 1. 모델 로드
model = load_model('./accuracy_85.keras')  # 모델 파일 경로

# 2. 이미지 전처리
def preprocess_image(img_path, target_size=(244, 244)):
    # 이미지 불러오기 및 크기 조정
    img = image.load_img(img_path, target_size=target_size)
    
    # 이미지를 numpy 배열로 변환하고 배치 차원 추가
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 정규화 (필요 시)
    img_array /= 255.0
    return img_array

# 3. 예측 수행
def predict_image(model, img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return predicted_class, predictions

# 이미지 경로 지정
img_path = './Classification_Image/식혜.jpg'

# 이미지 전처리 (244x244 크기)
img_array = preprocess_image(img_path)

# 예측 수행
predicted_class, predictions = predict_image(model, img_array)

# 결과 출력
print(f"Predicted class: {predicted_class}")
print(f"Prediction probabilities: {predictions}")


classlist = ["cans", "glass", "other_ps", "p_bowls", "pets"]
cnt = 0
predictions_list = predictions.tolist()
for i in predictions_list:
    for j in i:
        if(j != 1.0):
            cnt += 1
        else:
            print("예측 이미지 :", classlist[cnt])