import os
import time
from typing import Any, Callable, Dict, List
from google import genai
from google.genai import types
from openai import OpenAI

def retry_with_backoff(func: Callable, retries: int = 3, backoff_factor: float = 2.0) -> Any:
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(backoff_factor ** attempt)

def call_openai(prompt: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return {
        "provider": "OpenAI",
        "model": model,
        "content": response.choices[0].message.content,
        "latency": round(time.time() - start, 3),
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens
    }

def call_gemini(prompt: str, model: str = "gemini-3.8-flash") -> Dict[str, Any]:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(temperature=0.7)
    start = time.time()
    
    def _api_call():
        return client.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )
    
    response = retry_with_backoff(_api_call)
    return {
        "provider": "Gemini",
        "model": model,
        "content": response.text,
        "latency": round(time.time() - start, 3),
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count
    }

def compare_models(prompt: str):
    print(f"\n================ SO SÁNH CÁC MODEL ================")
    print(f"Prompt: '{prompt}'\n" + "-" * 50)
    
    # Test Gemini
    try:
        res = call_gemini(prompt)
        print(f"► Provider: {res['provider']} ({res['model']})")
        print(f"  Latency: {res['latency']}s | Tokens: Input={res['input_tokens']} | Output={res['output_tokens']}")
        print(f"  Nội dung: {res['content'][:150]}...\n")
    except Exception as e:
        print(f"Lỗi Gemini: {e}\n")

    # Test OpenAI
    try:
        res = call_openai(prompt)
        print(f"► Provider: {res['provider']} ({res['model']})")
        print(f"  Latency: {res['latency']}s | Tokens: Input={res['input_tokens']} | Output={res['output_tokens']}")
        print(f"  Nội dung: {res['content'][:150]}...\n")
    except Exception as e:
        print(f"► Provider: OpenAI | Tạm thời bỏ qua ({e})\n")

def streaming_chatbot():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history: List[Dict[str, str]] = []
    
    print("\n=== GEMINI STREAMING CHATBOT (Gõ 'exit' để thoát) ===")
    
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("Thoát chatbot.")
            break
            
        history.append({"role": "user", "content": user_input})
        recent_history = history[-6:]
        
        formatted_contents = [
            types.Content(
                role="user" if msg["role"] == "user" else "model",
                parts=[types.Part.from_text(text=msg["content"])]
            ) for msg in recent_history
        ]
        
        print("Bot: ", end="", flush=True)
        
        try:
            # Dùng chuẩn tên model gemini-3.8-flash
            response_stream = client.models.generate_content_stream(
                model="gemini-3.8-flash",
                contents=formatted_contents
            )
            
            full_response = ""
            for chunk in response_stream:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
            print()
            
            history.append({"role": "model", "content": full_response})
            
        except Exception as e:
            print(f"\n[⚠️ Lỗi ngắt kết nối/API]: {e}")
            print("Vui lòng thử gõ lại câu hỏi.")
            history.pop()

if __name__ == "__main__":
    compare_models("Giải thích khái niệm API là gì trong 2 câu ngắn gọn.")
    streaming_chatbot()