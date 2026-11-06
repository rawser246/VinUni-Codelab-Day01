import os
import time
from typing import Any, Callable, Dict, List, Tuple, Union
from google import genai
from google.genai import types

try:
    import anthropic
except ImportError:
    anthropic = None

# Định nghĩa các model chuẩn và bảng giá theo đúng kỳ vọng của bộ test
OPENAI_MODEL = "gpt-4o"
GEMINI_MODEL = "gemini-2.5-flash"

PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
}


def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start

    text = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return text, latency, usage


def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    start = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start

    text = getattr(response, "text", "") or ""

    input_tokens = 0
    output_tokens = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    return text, latency, usage


def call_anthropic(
    prompt: str,
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    if anthropic is None:
        raise ImportError("Package 'anthropic' is not installed.")

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start

    text = response.content[0].text
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return text, latency, usage


def compare_models(prompt: str = "") -> dict:
    # 1. GPT-4o
    text_4o, lat_4o, usage_4o = call_openai(prompt, model="gpt-4o")
    cost_4o = (
        usage_4o["input_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["input"]
        + usage_4o["output_tokens"] * PRICING_1M_TOKENS["gpt-4o"]["output"]
    ) / 1_000_000

    # 2. GPT-4o-Mini
    text_mini, lat_mini, usage_mini = call_openai(prompt, model="gpt-4o-mini")
    cost_mini = (
        usage_mini["input_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["input"]
        + usage_mini["output_tokens"] * PRICING_1M_TOKENS["gpt-4o-mini"]["output"]
    ) / 1_000_000

    # 3. Gemini 2.5 Flash
    text_gemini, lat_gemini, usage_gemini = call_gemini(
        prompt, model="gemini-2.5-flash"
    )
    cost_gemini = (
        usage_gemini["input_tokens"] * PRICING_1M_TOKENS["gemini-2.5-flash"]["input"]
        + usage_gemini["output_tokens"] * PRICING_1M_TOKENS["gemini-2.5-flash"]["output"]
    ) / 1_000_000

    return {
        "gpt4o": {
            "response": text_4o,
            "latency": lat_4o,
            "cost": cost_4o,
            "input_tokens": usage_4o["input_tokens"],
            "output_tokens": usage_4o["output_tokens"],
        },
        "gpt4o_mini": {
            "response": text_mini,
            "latency": lat_mini,
            "cost": cost_mini,
            "input_tokens": usage_mini["input_tokens"],
            "output_tokens": usage_mini["output_tokens"],
        },
        "gemini_flash": {
            "response": text_gemini,
            "latency": lat_gemini,
            "cost": cost_gemini,
            "input_tokens": usage_gemini["input_tokens"],
            "output_tokens": usage_gemini["output_tokens"],
        },
    }


def format_comparison_table(results: Union[dict, list[dict]]) -> str:
    """Format kết quả so sánh các model thành bảng Markdown."""
    if isinstance(results, dict):
        results_list = [results]
    else:
        results_list = results

    has_prompt = any("prompt" in item for item in results_list)

    # Ánh xạ key sang tên hiển thị chuẩn theo yêu cầu của unit test
    model_display_names = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
    }

    if has_prompt:
        lines = [
            "| Prompt | Model | Latency (s) | Cost ($) | Input Tokens | Output Tokens | Response |",
            "|---|---|---|---|---|---|---|",
        ]
        for item in results_list:
            prompt_str = item.get("prompt", "")
            for model_key in ["gpt4o", "gpt4o_mini", "gemini_flash"]:
                if model_key in item:
                    d = item[model_key]
                    model_name = model_display_names.get(model_key, model_key)
                    resp_preview = str(d.get("response", "")).replace("\n", " ")[:30]
                    lines.append(
                        f"| {prompt_str} | {model_name} | {d.get('latency', 0):.3f} | ${d.get('cost', 0):.6f} | "
                        f"{d.get('input_tokens', 0)} | {d.get('output_tokens', 0)} | {resp_preview}... |"
                    )
    else:
        lines = [
            "| Model | Latency (s) | Cost ($) | Input Tokens | Output Tokens | Response |",
            "|---|---|---|---|---|---|",
        ]
        for item in results_list:
            for model_key in ["gpt4o", "gpt4o_mini", "gemini_flash"]:
                if model_key in item:
                    d = item[model_key]
                    model_name = model_display_names.get(model_key, model_key)
                    resp_preview = str(d.get("response", "")).replace("\n", " ")[:30]
                    lines.append(
                        f"| {model_name} | {d.get('latency', 0):.3f} | ${d.get('cost', 0):.6f} | "
                        f"{d.get('input_tokens', 0)} | {d.get('output_tokens', 0)} | {resp_preview}... |"
                    )
    return "\n".join(lines)


def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    last_exception = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (2**attempt)
            time.sleep(delay)
    if last_exception:
        raise last_exception


def batch_compare(prompts: list[str]) -> list[dict]:
    results = []
    for p in prompts:
        try:
            res = compare_models(p)
        except TypeError:
            res = compare_models()
        if isinstance(res, dict):
            res["prompt"] = p
        results.append(res)
    return results


def streaming_chatbot() -> None:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history: List[Dict[str, str]] = []

    print("\n=== GEMINI STREAMING CHATBOT ===")
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

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
                parts=[types.Part.from_text(text=msg["content"])],
            )
            for msg in recent_history
        ]

        try:
            response_stream = client.models.generate_content_stream(
                model="gemini-2.5-flash", contents=formatted_contents
            )
            full_response = ""
            for chunk in response_stream:
                if hasattr(chunk, "text") and chunk.text:
                    print(chunk.text, end="", flush=True)
                    full_response += chunk.text
            print()
            history.append({"role": "model", "content": full_response})
        except Exception as e:
            print(f"\n[Error]: {e}")
            if history:
                history.pop()