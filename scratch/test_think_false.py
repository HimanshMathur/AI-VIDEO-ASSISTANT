import base64
import requests

with open('scratch/frames/frame_000_1s.jpg', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()

prompts = [
    "Describe what you see in this image concisely.",
    "How many people are visible in this video frame and what are they doing?",
    "What presentation slides, screen content, or text are visible in this video frame?",
]

for p_text in prompts:
    p = {
        'model': 'qwen3-vl:2b',
        'messages': [{'role': 'user', 'content': p_text, 'images': [b64]}],
        'stream': False,
        'think': False,
        'options': {'temperature': 0.1, 'num_predict': 350}
    }
    r = requests.post('http://localhost:11434/api/chat', json=p).json()
    msg = r.get('message', {})
    ans = msg.get('content', '')
    reasoning = msg.get('reasoning', '')
    print(f"\nPROMPT: {p_text}")
    print(f"REASONING ({len(reasoning)} chars): {reasoning[:80]}...")
    print(f"ANSWER ({len(ans)} chars): {ans.strip()}")
