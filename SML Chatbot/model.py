import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Device & dtype (fast + safe)
if torch.cuda.is_available():
    device_map = "auto"; torch_dtype = torch.float16
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device_map = "auto"; torch_dtype = torch.float16     # Apple Silicon
else:
    device_map = None;    torch_dtype = torch.float32     # CPU

print("Loading:", MODEL)
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, token=HF_TOKEN)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    token=HF_TOKEN,
    device_map=device_map,
    dtype=torch_dtype,
)

try:
    print("Resolved model:", model.config._name_or_path)
except Exception:
    pass



def build_prompt(messages):
    # model's chat template
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    prompt = ""
    sys_msgs = [m["content"] for m in messages if m["role"] == "system"]
    if sys_msgs:
        prompt += f"### System:\n{sys_msgs[-1]}\n\n"
    for m in messages:
        if m["role"] == "user":
            prompt += f"### Instruction:\n{m['content']}\n\n"
        elif m["role"] == "assistant":
            prompt += f"### Response:\n{m['content']}\n\n"
    prompt += "### Response:\n"
    return prompt

def generate_reply(history, user_msg, max_new_tokens=96, temperature=0.7, top_p=0.9):
    system_msg = {"role": "system", "content": "You are a friendly, fun friend."}
    msgs = [system_msg] + history + [{"role": "user", "content": user_msg}]
    prompt = build_prompt(msgs)

    inputs = tok(prompt, return_tensors="pt")
    if device_map:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    if "### Response:" in text:
        return text.split("### Response:")[-1].strip()
    return text.strip()




if __name__ == "__main__":
    print("Model loaded: Start chatting! (type 'exit' to quit)")
    history = []
    while True:
        user = input("\nYou: ").strip()
        if user.lower() in ("exit", "quit"):
            print("Bye!")
            break
        reply = generate_reply(history, user, max_new_tokens=96)
        print(f"Karren: {reply}")
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": reply})
