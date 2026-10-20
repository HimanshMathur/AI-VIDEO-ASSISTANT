import base64
import json
import requests

with open('scratch/frames/frame_000_1s.jpg', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()

prompt = (
    "Analyze this video frame concisely with observable facts only:\n"
    "1. Number of people visible and brief description:\n"
    "2. Visible actions or gestures:\n"
    "3. Screen content or slides shown:\n"
    "4. Legible text on screen/slides:\n"
    "5. Setting and environment:\n"
    "Do NOT guess or invent names or identities."
)

p = {
    'model': 'qwen3-vl:2b',
    'messages': [{'role': 'user', 'content': prompt, 'images': [b64]}],
    'stream': False,
    'options': {'temperature': 0.1, 'num_predict': 300}
}

r = requests.post('http://localhost:11434/api/chat', json=p).json()
msg = r.get('message', {})
content = msg.get('content')
print("Content length:", len(content) if content else 0)
print("Raw Content:\n", content)
