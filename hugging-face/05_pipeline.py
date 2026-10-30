"""
전처리 (Preprocessing): 사람의 글자(Text)를 컴퓨터가 아는 숫자(Token)로 변환
모델 추론 (Inference): 숫자를 딥러닝 모델에 넣어 결과 계산
후처리 (Postprocessing): 계산된 숫자를 다시 사람이 읽기 쉬운 텍스트(긍정/부정 등)로 변환
"""
# 1. Hugging Face의 pipeline 불러오기
from transformers import pipeline

# 2. 원하는 작업(task) 지정하여 pipeline 생성하기
# - task="text-classification"를 지정하면 관련 모델과 토크나이저를 알아서 세팅합니다.
classifier = pipeline(task="text-classification") # Huggingface : Text Classification

# 2-1. 로드된 모델의 이름(Hugging Face Hub 경로) 확인
model_name = classifier.model.name_or_path
print(f"사용된 기본 모델 이름: {model_name}")
# 2-2. 모델의 세부 구조/클래스 정보 확인
print(f"모델 클래스 타입: {type(classifier.model)}")

# 3. 문장을 넣어서 바로 결과 받아보기
result = classifier("오늘 Hugging Face를 공부하는데 정말 재미있다!")

# 4. 결과 출력
print(result)



