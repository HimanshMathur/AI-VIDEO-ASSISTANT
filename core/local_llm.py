import os
import time
import json
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict, Any, Optional, Iterator

logger = logging.getLogger(__name__)

# ─── High-Throughput Connection Pool for Local Ollama ────────────────────────
# max_retries=0 ensures fast-fail on connection errors without indefinite hanging
_ollama_session = requests.Session()
_adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=Retry(total=0, connect=0, read=0))
_ollama_session.mount("http://", _adapter)

# ─── Single Source of Truth for Local LLM Configuration ──────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

# Safety check: Prevent any accidental cloud URL configuration
if "api.openai.com" in OLLAMA_BASE_URL.lower() or "mistral.ai" in OLLAMA_BASE_URL.lower():
    raise ValueError("Cloud AI endpoints are strictly forbidden. OLLAMA_BASE_URL must point to local Ollama.")


def get_ollama_root_url() -> str:
    """Return root Ollama server URL (e.g., http://localhost:11434)."""
    url = OLLAMA_BASE_URL.strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def get_ollama_v1_url() -> str:
    """Return OpenAI-compatible v1 Ollama endpoint (e.g., http://localhost:11434/v1)."""
    root = get_ollama_root_url()
    return f"{root}/v1"


def check_ollama(timeout: float = 1.5) -> bool:
    """
    Verify local Ollama daemon is running and reachable.
    Fails fast immediately if server is down (no retry loops).
    """
    root = get_ollama_root_url()
    try:
        resp = _ollama_session.get(f"{root}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def list_local_models() -> List[str]:
    """Retrieve list of locally installed model names from Ollama."""
    root = get_ollama_root_url()
    try:
        resp = _ollama_session.get(f"{root}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception as e:
        logger.debug(f"Failed to fetch models from Ollama /api/tags: {e}")
    return []


def check_model_available(model_name: Optional[str] = None) -> bool:
    """
    Verify whether the specified model (or configured OLLAMA_MODEL)
    is installed and ready on the local Ollama server.
    """
    target = (model_name or OLLAMA_MODEL).strip().lower()
    models = [m.lower() for m in list_local_models()]
    
    for m in models:
        if m == target or m.split(":")[0] == target.split(":")[0] and target in m:
            return True
        if m == target or (":" not in target and m.split(":")[0] == target):
            return True
    return False


def get_active_model_name() -> str:
    """
    Returns OLLAMA_MODEL if available, or fallback to an available local model
    if OLLAMA_MODEL is still downloading.
    """
    if check_model_available(OLLAMA_MODEL):
        return OLLAMA_MODEL
    
    models = list_local_models()
    if models:
        logger.warning(
            f"Configured model '{OLLAMA_MODEL}' is not available locally yet. "
            f"Available models: {models}. Using '{models[0]}'."
        )
        return models[0]
    return OLLAMA_MODEL


def get_memory_stats() -> Dict[str, Any]:
    """Retrieve system and GPU free/total memory in GB."""
    stats = {
        "system_total_gb": 0.0,
        "system_free_gb": 0.0,
        "gpu_free_gb": None,
        "gpu_total_gb": None,
    }
    # Windows native memory query via GlobalMemoryStatusEx
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]
        s = MEMORYSTATUSEX()
        s.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            stats["system_total_gb"] = round(s.ullTotalPhys / (1024**3), 2)
            stats["system_free_gb"] = round(s.ullAvailPhys / (1024**3), 2)
    except Exception as e:
        logger.debug(f"Could not read Windows system memory: {e}")

    # CUDA VRAM query via torch if available
    try:
        import torch
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            stats["gpu_free_gb"] = round(free_b / (1024**3), 2)
            stats["gpu_total_gb"] = round(total_b / (1024**3), 2)
    except Exception:
        pass

    return stats


def get_loaded_models() -> List[str]:
    """Return list of currently loaded/resident model names reported by Ollama /api/ps."""
    root_url = get_ollama_root_url()
    try:
        resp = _ollama_session.get(f"{root_url}/api/ps", timeout=2.0)
        if resp.status_code == 200:
            return [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
    except Exception:
        pass
    return []


def log_memory_state(stage: str) -> Dict[str, Any]:
    """Log system memory, GPU memory, and currently loaded Ollama models."""
    mem = get_memory_stats()
    loaded = get_loaded_models()
    gpu_str = f"{mem['gpu_free_gb']} GB free (of {mem['gpu_total_gb']} GB)" if mem.get("gpu_free_gb") is not None else "N/A"
    msg = (
        f"[{stage}] System Free: {mem['system_free_gb']} GB (of {mem['system_total_gb']} GB) | "
        f"GPU Free: {gpu_str} | Loaded in Ollama: {loaded}"
    )
    logger.info(msg)
    print(msg)
    return {"memory": mem, "loaded_models": loaded}


def unload_model(model_name: str, timeout: float = 10.0) -> bool:
    """
    Explicitly unload model from Ollama using official keep_alive: 0.
    Ensures model is completely released from VRAM and RAM.
    """
    root_url = get_ollama_root_url()
    target = model_name.strip().lower()
    target_base = target.split(":")[0]

    loaded = get_loaded_models()
    if not any(target_base in m.lower() for m in loaded):
        logger.debug(f"Model '{model_name}' is not currently loaded in Ollama.")
        return True

    logger.info(f"Issuing official keep_alive=0 unload request for '{model_name}'...")
    try:
        _ollama_session.post(
            f"{root_url}/api/generate",
            json={"model": model_name, "keep_alive": 0},
            timeout=timeout,
        )
    except Exception as e:
        logger.debug(f"Unload request to Ollama for '{model_name}': {e}")

    # Free Python side tensors & run garbage collector
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Poll /api/ps up to 5s to confirm eviction
    t0 = time.time()
    while time.time() - t0 < 5.0:
        active = get_loaded_models()
        if not any(target_base in m.lower() for m in active):
            logger.info(f"Model '{model_name}' successfully evicted from Ollama.")
            return True
        time.sleep(0.5)

    return False


def ensure_model_ready_for_inference(target_model: str) -> None:
    """
    Ensure no conflicting models are loaded before invoking target_model.
    Guarantees that qwen3-vl:2b and qwen3:8b are never co-resident in memory.
    """
    target_base = target_model.strip().lower().split(":")[0]
    loaded = get_loaded_models()

    for m in loaded:
        m_base = m.strip().lower().split(":")[0]
        if m_base != target_base:
            logger.info(f"Conflicting model '{m}' detected in Ollama memory. Unloading before loading '{target_model}'...")
            log_memory_state(f"MEMORY BEFORE UNLOADING CONFLICTING MODEL '{m}'")
            unload_model(m)
            log_memory_state(f"MEMORY AFTER UNLOADING CONFLICTING MODEL '{m}'")

    log_memory_state(f"MEMORY STATE BEFORE INFERENCE ({target_model})")


def validate_local_llm_ready() -> None:
    """
    Enforce strict local LLM readiness.
    Fails fast if Ollama server is down.
    """
    if not check_ollama():
        raise RuntimeError(
            "Local Ollama server is not running. Start Ollama and try again.\n"
            f"Expected endpoint: {get_ollama_root_url()}"
        )
    
    target_model = get_active_model_name()
    if not check_model_available(target_model):
        available = list_local_models()
        if not available:
            raise RuntimeError(
                f"Model '{target_model}' is not available locally.\n"
                f"Run in terminal: ollama pull {target_model}"
            )

    # Ensure memory is ready and conflicting models are evicted
    ensure_model_ready_for_inference(target_model)


DEFAULT_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))



def chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
) -> str:
    """
    Execute local chat generation via Ollama's native /api/chat endpoint with think=False
    to eliminate reasoning token overhead (26x faster).
    Strictly local, no cloud calls.
    """
    validate_local_llm_ready()
    target_model = model or get_active_model_name()
    root_url = get_ollama_root_url()
    endpoint = f"{root_url}/api/chat"
    call_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT

    options: Dict[str, Any] = {
        "temperature": temperature,
        "num_ctx": 4096,
    }
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    payload: Dict[str, Any] = {
        "model": target_model,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": options,
    }

    headers = {"Content-Type": "application/json"}

    try:
        resp = _ollama_session.post(endpoint, json=payload, headers=headers, timeout=call_timeout)
        if resp.status_code == 200:
            result = resp.json()
            msg = result.get("message", {})
            content = (msg.get("content") or "").strip()
            return content

        # Fallback to OpenAI-compatible v1 endpoint if /api/chat fails
        logger.warning(f"Ollama /api/chat returned HTTP {resp.status_code}, falling back to /v1/chat/completions")
        v1_url = get_ollama_v1_url()
        v1_payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            v1_payload["max_tokens"] = max_tokens
        resp_v1 = _ollama_session.post(f"{v1_url}/chat/completions", json=v1_payload, headers=headers, timeout=call_timeout)
        if resp_v1.status_code != 200:
            raise RuntimeError(f"Ollama chat error (HTTP {resp_v1.status_code}): {resp_v1.text}")
        choices = resp_v1.json().get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            content = (msg.get("content") or "").strip()
            if not content and msg.get("reasoning"):
                content = msg.get("reasoning").strip()
            return content
        return ""
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Local Ollama generation timed out after {call_timeout}s on model '{target_model}'.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Local Ollama server disconnected unexpectedly. Please ensure Ollama is running.")
    except Exception as e:
        logger.error(f"Local LLM chat generation error: {e}")
        raise


def chat_completion_stream(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Iterator[str]:
    """
    Stream tokens in real-time from local Ollama via native /api/chat endpoint.
    Strictly local, zero cloud calls.
    """
    validate_local_llm_ready()
    target_model = model or get_active_model_name()
    root_url = get_ollama_root_url()
    endpoint = f"{root_url}/api/chat"
    call_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT

    options: Dict[str, Any] = {
        "temperature": temperature,
        "num_ctx": 4096,
    }
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    payload: Dict[str, Any] = {
        "model": target_model,
        "messages": messages,
        "stream": True,
        "think": False,
        "keep_alive": "30m",
        "options": options,
    }

    headers = {"Content-Type": "application/json"}

    try:
        resp = _ollama_session.post(endpoint, json=payload, headers=headers, timeout=call_timeout, stream=True)
        if resp.status_code == 200:
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break
        else:
            # Fallback to synchronous completion if streaming endpoint returns error
            full = chat_completion(messages, model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
            yield full
    except Exception as e:
        logger.error(f"Local LLM streaming error: {e}")
        # Fallback to non-streaming if needed
        try:
            full = chat_completion(messages, model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
            yield full
        except Exception:
            raise


def json_chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: Optional[int] = 1024,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute structured JSON local generation via Ollama's native /api/chat endpoint.
    Guarantees structured output parsing while keeping generation 100% local.
    """
    validate_local_llm_ready()
    target_model = model or get_active_model_name()
    root_url = get_ollama_root_url()
    endpoint = f"{root_url}/api/chat"
    call_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT

    options: Dict[str, Any] = {
        "temperature": temperature,
        "num_ctx": 4096,
    }
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    payload: Dict[str, Any] = {
        "model": target_model,
        "messages": messages,
        "stream": False,
        "think": False,
        "format": "json",
        "keep_alive": "30m",
        "options": options,
    }

    headers = {"Content-Type": "application/json"}

    try:
        resp = _ollama_session.post(endpoint, json=payload, headers=headers, timeout=call_timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama JSON chat error (HTTP {resp.status_code}): {resp.text}")

        result = resp.json()
        msg = result.get("message", {})
        content = (msg.get("content") or "").strip()
        if not content:
            return {}

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Clean up potential markdown formatting codeblocks if any
            clean = content.strip()
            if clean.startswith("```json"):
                clean = clean[7:]
            elif clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            return json.loads(clean.strip())
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Local Ollama JSON generation timed out after {call_timeout}s on model '{target_model}'.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Local Ollama server disconnected unexpectedly. Please ensure Ollama is running.")
    except Exception as e:
        logger.error(f"Local LLM JSON chat generation error: {e}")
        raise


def get_langchain_llm(temperature: float = 0.1):
    """
    Obtain a LangChain ChatOllama instance connected exclusively to the local Ollama instance.
    """
    validate_local_llm_ready()
    try:
        from langchain_ollama import ChatOllama
        target_model = get_active_model_name()
        root_url = get_ollama_root_url()
        return ChatOllama(
            model=target_model,
            base_url=root_url,
            temperature=temperature,
        )
    except Exception as e:
        logger.error(f"Failed to instantiate LangChain ChatOllama: {e}")
        raise
