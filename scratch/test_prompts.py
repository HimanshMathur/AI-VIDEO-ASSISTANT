import base64
import requests

with open('scratch/frames/frame_000_1s.jpg', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()

prompts = [
    "Describe this video frame concisely: who is visible, what are they doing, and what is on the screen?",
    "How many people are visible in this image and what are they doing?",
    "What text or slides are visible on the screen in this video frame?",
]

for p_text in prompts:
    p = {
        'model': 'qwen3-vl:2b',
        'messages': [{'role': 'user', 'content': p_text, 'images': [b64]}],
        'stream': False,
        'options': {'temperature': 0.1, 'num_predict': 256}
    }
    r = requests.post('http://localhost:11434/api/chat', json=p).json()
    ans = r.get('message', {}).get('content', '')
    print(f"\nPROMPT: {p_text}")
    print(f"ANSWER: {ans.strip()}")
