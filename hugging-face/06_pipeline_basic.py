from transformers import pipeline
import os

# Windows 경고 메시지 지우기 (선택 사항)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 텍스트분류
clf = pipeline(task="text-classification")
print(clf('Hugging Face is Amazing!!'))

# 질의응답 - model 명이 확실 하다면 굳이 task 를 넣을 필요가 없다.
qa = pipeline(model="monologg/koelectra-base-v3-finetuned-korquad")
print(qa(
    question="대한민국의 수도는?",
    context="대한민국의 수도는 서울 입니다."
))

# 이미지분류 - 이미지 인식을 위한 라이브러리 설치 필요
# pip install pillow torchvision
img = pipeline(model="google/vit-base-patch16-224")
print(img("dog.png"))
