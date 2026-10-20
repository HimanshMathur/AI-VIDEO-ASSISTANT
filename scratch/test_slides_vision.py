import base64
import requests

test_frames = [
    ('Meeting Grid', 'scratch/frames/frame_000_1s.jpg'),
    ('Slide Frame 1', 'scratch/frames/frame_004_539s.jpg'),
    ('Slide Frame 2', 'scratch/frames/frame_015_2020s.jpg'),
]

prompt = (
    "Analyze this video frame factually and concisely:\n"
    "1. People count and participant layout:\n"
    "2. Slide / screen title or visible headings:\n"
    "3. Bullet points or legible text shown on screen:\n"
    "4. Presenter or active visual activity:"
)

for label, path in test_frames:
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    
    p = {
        'model': 'qwen3-vl:2b',
        'messages': [{'role': 'user', 'content': prompt, 'images': [b64]}],
        'stream': False,
        'options': {'temperature': 0.1, 'num_predict': 600}
    }
    r = requests.post('http://localhost:11434/api/chat', json=p).json()
    msg = r.get('message', {})
    content = msg.get('content', '').strip()
    thinking = msg.get('thinking', '').strip()
    
    # If content was cut off by length or empty, fallback to rich thinking
    text = content if content else thinking
    print(f"==================================================")
    print(f"FRAME: {label} ({path})")
    print(f"Content length: {len(content)} | Thinking length: {len(thinking)}")
    print(f"EXTRACTED TEXT:\n{text}\n")
