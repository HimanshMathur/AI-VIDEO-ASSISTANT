import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
os.environ.setdefault('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
os.environ.setdefault('OLLAMA_MODEL', 'qwen3:8b')
os.environ.setdefault('WHISPER_MODEL', 'small')
os.environ.setdefault('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
os.environ.setdefault('VISION_MODEL', 'qwen2.5vl:3b')

# Test 1: compile all python files
import compileall
print("=== Compile check ===")
ok = compileall.compile_dir('.', quiet=True, force=True)
print(f"compile_dir result: {'OK (all files compiled)' if ok else 'ERRORS found'}")

print()
print("=== Module import check ===")
mods = [
    'utils.audio_processor',
    'core.local_llm',
    'core.extractor',
    'core.summarizer',
    'core.vector_store',
    'core.rag_engine',
    'core.visual_analyzer',
    'core.transcriber',
]
errors = []
for m in mods:
    try:
        __import__(m)
        print(f'  OK  {m}')
    except Exception as e:
        print(f'  FAIL {m}: {type(e).__name__}: {e}')
        errors.append((m, str(e)))

print()
print(f"=== Result: {len(errors)} import errors ===")
if errors:
    for m, e in errors:
        print(f"  {m}: {e}")

# Test 2: Check specific things
print()
print("=== Specific checks ===")

# Check visual_ai in DEFAULT_STATE
code = open('app.py', encoding='utf-8').read()
has_visual_ai = '"visual_ai": "pending"' in code
print(f"visual_ai in DEFAULT_STATE: {has_visual_ai}")

# Check no top-level whisper import
transcriber_code = open('core/transcriber.py', encoding='utf-8').read()
lines_with_whisper = [l.strip() for l in transcriber_code.split('\n') if 'import whisper' in l and not l.strip().startswith('#')]
top_level_whisper = [l for l in lines_with_whisper if not l.startswith('import whisper  # lazy')]
print(f"Top-level 'import whisper' removed: {len(top_level_whisper) == 0}")
print(f"  Lines with 'import whisper': {lines_with_whisper}")

# Check HF offline fix in vector_store
vs_code = open('core/vector_store.py', encoding='utf-8').read()
offline_before_try = vs_code.find('HF_HUB_OFFLINE') < vs_code.find('local_files_only=True')
print(f"HF_HUB_OFFLINE set BEFORE local_files_only (should be False): {offline_before_try}")

# Check walrus operator removed
proc_code = open('utils/audio_processor.py', encoding='utf-8').read()
has_walrus = 'clean_url_or_path :=' in proc_code
print(f"Walrus dead code removed (should be False): {has_walrus}")
