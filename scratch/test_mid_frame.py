import base64
import requests

with open('scratch/frames/frame_006_808s.jpg', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()

prompt = (
    "Describe this video frame factually and concisely:\n"
    "- Number of visible people and what they are doing\n"
    "- Any shared screen, presentation slides, diagrams, or documents shown\n"
    "- Legible text or titles on screen\n"
    "Do NOT guess or invent names."
)

p = {
    'model': 'qwen3-vl:2b',
    'messages': [{'role': 'user', 'content': prompt, 'images': [b64]}],
    'stream': False,
    'think': False,
    'options': {'temperature': 0.1, 'num_predict': 350}
}

r = requests.post('http://localhost:11434/api/chat', json=p).json()
print("Middle Frame (808s) Analysis:\n", r.get('message', {}).get('content', ''))
