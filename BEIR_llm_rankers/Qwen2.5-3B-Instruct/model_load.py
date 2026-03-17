# test_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

MODEL_PATH = "/mnt/f/Ubuntu/models/Qwen2.5-3B-Instruct"

print("=" * 40)
print("Step 1/3：加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("✅ Tokenizer 加载成功")

print("\nStep 2/3：加载模型（4bit 量化）...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()
print("✅ 模型加载成功")
print(f"   设备分配：{model.hf_device_map}")

print("\nStep 3/3：推理测试...")
messages = [{"role": "user", "content": "Say hello in one word."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
print(f"✅ 推理成功")
print(f"   模型输出：{response}")
print("=" * 40)
print("🎉 所有测试通过，可以开始重排实验")