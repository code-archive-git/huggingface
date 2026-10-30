# accelerate(Hugging Face Accelerate 라이브러리)
# 딥러닝 모델을 여러 대의 GPU(Multi-GPU)나 CPU, 다양한 하드웨어 장치에서 쉽고 효율적으로 실행할 수 있도록 도와주는 도구
# pip install accelerate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2-2b-it"
# 1. 토크나이저 생성
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. 대화 내용을 정의
msg_list = [
    {"role":"user","content":"파이썬에서 변수가 뭐야?"},
    {"role": "assistant", "content": "변수는 데이터를 저장하는 상자와 같습니다."},
    {"role": "user", "content": "그럼 리스트는 뭐야?"},
]
# 3. 대답할 내용 토큰 화 테스트
prompt = tokenizer.apply_chat_template(
    msg_list,
    tokenize = False, # False : 토큰화 된 내용을 문자열로 반환
    add_generation_prompt=True, # True : msg_list 이후 assistant 가 이어 쓸수 있을지 여부
)

print(f'모델에 입력될 최종 텍스트 포맷 : {prompt}')

#### 대답을 듣기 위한 작업 ###
# 4. 모델 생성
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float32,
    device_map="auto"
)

# 5. 토큰의 ids 화(숫자화)
inputs = tokenizer(prompt,return_tensors="pt").to("cuda")
"""
inputs = tokenizer.apply_chat_template(
    msg_list,
    tokenize = True, # False : 토큰화 된 내용을 문자열로 반환
    add_generation_prompt=True, # True : msg_list 이후 assistant 가 이어 쓸수 있을지 여부
    return_tensors="pt",
).to('cuda')# 이 부분은 tensor 형태에만 적용 가능하다.
"""
print(f'{prompt} \n\n {inputs}')

# 6. 답변 생성
# 추론할 때는 기울기(gradient)를 계산할 필요가 없으므로
print("#### 답변 생성 ####")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,  # 생성할 최대 토큰 수
        temperature=0.7,  # 답변의 창의성 조절 (0.0: 정직함 ~ 1.0: 창의적)
        do_sample=True
    )


# outputs 에는 질문내용과 답변내용이 모두 들어있다.
generated_tokens = outputs[0]
"""
input_length = inputs.shape[1] # 입력내용 만큼을 측정해서 
generated_tokens = outputs[0][input_length:] # 빼고
"""
response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print("--- [2] AI의 최종 답변 ---")
print(response_text)
print("==================================================")