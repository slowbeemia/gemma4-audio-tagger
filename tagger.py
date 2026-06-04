#!/usr/bin/env python3
"""
gemma4-audio-tagger — Batch audio tagging via Gemma 4 12B (HuggingFace Transformers).

Uses the OFFICIAL Google-recommended API: AutoModelForMultimodalLM + AutoProcessor.
Audio is passed via the chat template — no manual encoding needed.

Usage:
    uv run python3 tagger.py                  # tag all untagged audio
    uv run python3 tagger.py --max 10         # tag 10 files, then stop
    uv run python3 tagger.py --max 1          # test one file
    uv run python3 tagger.py --dry-run        # show what would be tagged
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, pipeline

# --- Monkeypatch for Gemma 4 Unified Multimodal Embedder 4-bit Quantization Bug ---
try:
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedMultimodalEmbedder
    _old_forward = Gemma4UnifiedMultimodalEmbedder.forward

    def _patched_forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # If bitsandbytes 4-bit quantization is active, .weight.dtype is torch.uint8.
        # We must cast to compute_dtype (or bfloat16) instead of uint8!
        target_dtype = self.embedding_projection.weight.dtype
        if target_dtype == torch.uint8:
            target_dtype = getattr(self.embedding_projection, "compute_dtype", torch.bfloat16)
        inputs_embeds = inputs_embeds.to(target_dtype)
        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(embs_normed)

    Gemma4UnifiedMultimodalEmbedder.forward = _patched_forward
    print("[Patched] Gemma4UnifiedMultimodalEmbedder.forward monkeypatched successfully.", flush=True)
except Exception as e:
    print(f"Warning: Could not apply Gemma 4 Multimodal patch: {e}", flush=True)

# --- Config ----------------------------------------------------------------

_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)

ASSET_MANAGER_URL = os.getenv("ASSET_MANAGER_URL", "http://bob.fritz.box:8899")
DELAY_SEC = float(os.getenv("DELAY_SEC", "2"))
API_PAGE_SIZE = int(os.getenv("API_PAGE_SIZE", "50"))
PATH_MAP_FROM = os.getenv("PATH_MAP_FROM", "//orange/Assets")
PATH_MAP_TO = os.getenv("PATH_MAP_TO", "/mnt/orange")

MODEL_ID = "google/gemma-4-12B-it"

STATE_FILE = Path(__file__).parent / ".tagger_state.json"
LOG_FILE = Path(__file__).parent / "tagger.log"

# --- Audio prompt ----------------------------------------------------------

SYSTEM_PROMPT = """You are an audio analysis expert for game development. Listen carefully to the audio and respond ONLY with a JSON object:

{"tags": "tag1, tag2, tag3, tag4, tag5, tag6", "description": "Kurze deutsche Beschreibung des Sounds (1 Satz, game-dev-relevant)"}

Tags should be game-dev relevant categories (e.g. explosion, coin_loot, power_up, laser, ui_click, ambient_dark, magic_spell, footsteps, etc.).
Output ONLY the JSON, no other text."""

USER_PROMPT = """Analyze this audio clip for a game asset library. What kind of sound is this? Give 5-8 precise tags and a short German description."""


def log(msg: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def fetch_untagged_assets(client: httpx.Client, page_size: int = 50) -> list[dict]:
    """Fetch ALL untagged audio assets from Asset Manager API (paginated)."""
    all_assets = []
    offset = 0
    while True:
        try:
            resp = client.get(
                f"{ASSET_MANAGER_URL}/api/audio-assets/untagged",
                params={"limit": page_size, "offset": offset},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("assets", [])
            total = data.get("total", 0)
            all_assets.extend(batch)
            print(f"  ... fetched {len(all_assets)}/{total} from Asset Manager", flush=True)
            offset += page_size
            if offset >= total:
                break
        except httpx.HTTPError as e:
            log(f"ERROR fetching assets at offset {offset}: {e}")
            break

    return all_assets


def tag_audio(full_path: str, filename: str, rel_path: str) -> dict | None:
    """Send audio to Gemma 4 12B via HuggingFace Transformers."""
    local_path = full_path.replace(PATH_MAP_FROM, PATH_MAP_TO, 1)
    path = Path(local_path)
    if not path.exists():
        log(f"  SKIP: file not found: {local_path}")
        return None

    # Pass the local file path string directly — pipeline handles loading and resampling
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "audio", "audio": local_path},
            ],
        },
    ]

    try:
        outputs = pipe(
            messages,
            return_full_text=False,
            generate_kwargs={
                "max_new_tokens": 300,
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64
            }
        )
        content = outputs[0]["generated_text"]

        # Strip markdown fences
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        return json.loads(content)

    except Exception as e:
        log(f"  ERROR: inference failed: {e}")
        return None


def send_to_asset_manager(asset_id: int, tags: dict, client: httpx.Client) -> bool:
    """Send tags to Asset Manager API."""
    try:
        resp = client.post(
            f"{ASSET_MANAGER_URL}/api/audio-assets/{asset_id}/tag",
            json={
                "tags": tags.get("tags", ""),
                "description": tags.get("description", ""),
                "notes": "gemma4-12b",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        log(f"  ERROR: Asset Manager API failed: {e}")
        return False


def report_status(client: httpx.Client, state: dict, status: str = "running") -> None:
    try:
        client.post(
            f"{ASSET_MANAGER_URL}/api/tagging-status",
            json={
                "host": "thegrey-gemma4",
                "model": MODEL_ID,
                "version": "1.0",
                "status": status,
                "current": state["tagged"],
                "total": state["total"],
                "errors": state["errors"],
                "runtime_s": state.get("runtime_s", 0),
            },
            timeout=5,
        )
    except Exception:
        pass


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"tagged": 0, "errors": 0, "processed": [], "total": 0, "runtime_s": 0}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gemma 4 12B Audio Tagger")
    parser.add_argument("--max", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    log("=" * 60)
    log("Gemma 4 12B Audio Tagger (Transformers)")
    log(f"Model: {MODEL_ID}")
    log(f"Asset Manager: {ASSET_MANAGER_URL}")

    # --- Load model (shared across all inferences) ---
    global processor, model, pipe

    log("Loading model (4-bit, this takes ~2 min first time)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    log(f"Model loaded on: {model.device}")

    log("Initializing multimodal any-to-any pipeline...")
    pipe = pipeline("any-to-any", model=model, processor=processor)

    # --- Fetch untagged assets ---
    mgr = httpx.Client(timeout=httpx.Timeout(30))

    log("Fetching untagged audio from Asset Manager...")
    assets = fetch_untagged_assets(mgr, API_PAGE_SIZE)

    if args.dry_run:
        log(f"Dry-run: {len(assets)} untagged audio assets would be tagged")
        for a in assets[:20]:
            print(f"  #{a['id']}  {a['filename']}  ({a.get('duration', 0):.1f}s)  {a['rel_path']}")
        if len(assets) > 20:
            print(f"  ... and {len(assets) - 20} more")
        return

    state = load_state()
    if args.reset:
        state = {"tagged": 0, "errors": 0, "processed": [], "total": 0, "runtime_s": 0}
    processed = set(state.get("processed", []))
    state["total"] = len(assets)
    start_time = time.time()

    report_status(mgr, state, "running")

    tagged = 0
    for i, asset in enumerate(assets):
        asset_id = asset["id"]
        filename = asset["filename"]
        full_path = asset["full_path"]
        rel_path = asset.get("rel_path", filename)

        if str(asset_id) in processed:
            continue

        limit_msg = f" (--max={args.max})" if args.max else ""
        log(f"[{tagged + 1}/{min(state['total'], args.max or state['total'])}{limit_msg}] #{asset_id} {filename}")

        result = tag_audio(full_path, filename, rel_path)
        if result:
            if send_to_asset_manager(asset_id, result, mgr):
                tagged += 1
                state["tagged"] += 1
                log(f"  ✓ {result.get('tags', '')[:100]}")
                if result.get("description"):
                    log(f"    {result['description'][:120]}")
        else:
            state["errors"] += 1

        processed.add(str(asset_id))
        state["processed"] = list(processed)
        state["runtime_s"] = int(time.time() - start_time)

        if tagged % 10 == 0:
            save_state(state)
            report_status(mgr, state, "running")

        if args.max and tagged >= args.max:
            log(f"Reached --max={args.max}, stopping.")
            break

        time.sleep(DELAY_SEC)

    state["runtime_s"] = int(time.time() - start_time)
    save_state(state)
    report_status(mgr, state, "completed")

    log(f"Done! Tagged: {state['tagged']}, Errors: {state['errors']}, Runtime: {state['runtime_s']}s")


if __name__ == "__main__":
    main()
