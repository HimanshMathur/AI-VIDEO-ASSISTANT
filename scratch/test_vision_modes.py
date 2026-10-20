import base64
import requests

with open('scratch/frames/frame_000_1s.jpg', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()

configs = [
    {
        'name': 'think_false_flag',
        'body': {
            'model': 'qwen3-vl:2b',
            'messages': [{'role': 'user', 'content': 'Describe this image concisely.', 'images': [b64]}],
            'stream': False,
            'think': False,
            'options': {'num_predict': 150}
        }
    },
    {
        'name': 'num_predict_800',
        'body': {
            'model': 'qwen3-vl:2b',
            'messages': [{'role': 'user', 'content': 'Describe this image concisely in 2 sentences.', 'images': [b64]}],
            'stream': False,
            'options': {'temperature': 0.1, 'num_predict': 800}
        }
    }
]

for c in configs:
    r = requests.post('http://localhost:11434/api/chat', json=c['body']).json()
    msg = r.get('message', {})
    content = msg.get('content', '')
    thinking = msg.get('thinking', '')
    done_reason = r.get('done_reason', '')
    print(f"=== {c['name']} (done_reason: {done_reason}) ===")
    print(f"content len: {len(content)} | thinking len: {len(thinking)}")
    if content:
        print("CONTENT:\n", content.strip()[:300])
    if thinking:
        print("THINKING:\n", thinking.strip()[:300])
    print()
