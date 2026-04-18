from llama_cpp import Llama
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pocket_agent_q4.gguf")


llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

def run(prompt: str, history: list[dict]) -> str:
   
    messages = history + [{"role": "user", "content": prompt}]
    
   
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        stop=["</tool_call>", "<|endoftext|>", "USER:", "ASSISTANT:"],
        temperature=0.1
    )
    
    output = response["choices"][0]["message"]["content"].strip()
    
    if "<tool_call>" in output and "</tool_call>" not in output:
        output += "</tool_call>"
        
    return output