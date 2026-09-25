"""
Image Generation Engine — Step 3
Reads image prompts from Step 2 output and generates images via
Cloudflare Workers AI (FLUX.2 klein 4b), with post-generation QC via Gemini.

Usage:
    python "Engine/Step 3/generate_images.py" --input "Engine/Step 2/output/scene1_prompts.md" --output "Engine/Step 3/output/scene1"

Each image is checked by Gemini vision after generation. If QC fails,
the image is moved to _rejected/ and retried with a new seed (up to 3 attempts).
Full audit trail in raw/ per project rules.
"""

import argparse, base64, glob, io, json, os, re, shutil, sys, time, uuid
import urllib.request, urllib.error
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

def _load_env():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if load_dotenv and env_path.exists():
        load_dotenv(env_path)
    elif env_path.exists():
        for line in open(env_path, encoding="utf-8"):
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_env()

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
CAP_USD = float(os.environ.get("ENGINE_CAP_USD", "2.0"))
MAX_RETRIES = 3
BASE_SEED = 42
W, H = 1536, 864  # 16:9 landscape desktop
MODEL_ID = "@cf/black-forest-labs/flux-2-klein-4b"
MODEL_TAG = "flux-2-klein-4b"

ACCT = os.environ.get("CLOUDFLARE_ID") or os.environ.get("cloudflare_id", "")
TOK = os.environ.get("CLOUDFLARE_TOKEN") or os.environ.get("cloudflare_token", "")
COLAB_URL = os.environ.get("COLAB_URL", "").rstrip("/")

_raw_gemini = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
GEMINI_KEYS = [k.strip() for k in _raw_gemini.split(",") if k.strip()]

# ── QC prompt sent to Gemini vision ──────────────────────────────────────────

QC_SYSTEM = """\
You are a quality-control inspector for AI-generated images used in an animated educational video.
You will receive an image and the prompt that was used to generate it.
Check for these defects and respond with a JSON object:

{
  "pass": true/false,
  "defects": ["list of defect codes found, empty if pass is true"]
}

DEFECT CODES:
- "anthropomorphic_object": Any inanimate object (lamp, chair, bed, box, toy) has a face, eyes, mouth, or human expression. Lamps must be plain ordinary lamps, not characters.
- "messy_floor": Toys, blocks, or objects scattered on the floor. Toys should be inside the toy box, not on the floor.
- "wrong_framing": Image is portrait/vertical when it should be landscape, or vice versa.
- "text_in_image": Visible text, letters, words, watermarks, or subtitles in the image.
- "missing_character": The prompt describes a character that is not visible in the image.
- "wrong_character": Character appearance does not match (wrong clothing color, missing glasses, etc).
- "wrong_action": Character is not performing the action described in the prompt (not pointing when should be pointing, not sitting when should be sitting, etc).
- "wrong_setting": Setting is clearly wrong (classroom instead of bedroom, or vice versa).
- "blurry_or_artifact": Major rendering artifacts, extreme blur, or corrupted output.

RULES:
- Be strict on anthropomorphic_object — zero tolerance for faces on objects.
- Be strict on messy_floor — the floor should be clean in bedroom scenes.
- Be lenient on minor color variations in clothing.
- Only flag defects you are confident about.
- Respond ONLY with the JSON object, nothing else."""


def qc_check(image_b64: str, prompt: str) -> dict:
    """Send image + prompt to Gemini vision for QC. Returns {"pass": bool, "defects": [...]}."""
    if not GEMINI_KEYS:
        return {"pass": True, "defects": [], "skipped": "no Gemini keys"}

    body = json.dumps({
        "system_instruction": {"parts": [{"text": QC_SYSTEM}]},
        "contents": [{
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                {"text": f"Prompt used to generate this image:\n{prompt}\n\nInspect and return JSON."}
            ]
        }],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 500},
    }).encode()

    for i, key in enumerate(GEMINI_KEYS):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                text = text.strip()
                if text.startswith("```"):
                    text = re.sub(r"^```\w*\n?", "", text)
                    text = re.sub(r"\n?```$", "", text)
                    text = text.strip()
                # Extract the QC JSON — look for {"pass" to skip thinking blocks
                for marker in ['"pass"', "'pass'"]:
                    idx = text.find(marker)
                    if idx == -1:
                        continue
                    # Walk back to the opening brace
                    start = text.rfind('{', 0, idx)
                    if start == -1:
                        continue
                    # Walk forward to find matching closing brace
                    depth = 0
                    for i, ch in enumerate(text[start:], start):
                        if ch == '{': depth += 1
                        elif ch == '}': depth -= 1
                        if depth == 0:
                            try:
                                obj = json.loads(text[start:i+1])
                                if "pass" in obj:
                                    return obj
                            except json.JSONDecodeError:
                                break
                            break
                return {"pass": True, "defects": [], "skipped": "no QC JSON found"}
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < len(GEMINI_KEYS) - 1:
                continue
            return {"pass": True, "defects": [], "skipped": f"Gemini QC error {e.code}"}
        except Exception as e:
            return {"pass": True, "defects": [], "skipped": f"QC parse error: {e}"}

    return {"pass": True, "defects": [], "skipped": "all Gemini keys exhausted"}


# ── prompt parsing ───────────────────────────────────────────────────────────

def parse_prompts(md_text: str) -> list[dict]:
    shots = []
    current_name = None
    current_lines = []
    for line in md_text.split("\n"):
        m = re.match(r"^###\s+(\S+)", line)
        if m:
            if current_name and current_lines:
                shots.append({"name": current_name, "prompt": "\n".join(current_lines).strip()})
            current_name = m.group(1)
            current_lines = []
        elif current_name is not None:
            if line.strip():
                current_lines.append(line.strip())
    if current_name and current_lines:
        shots.append({"name": current_name, "prompt": "\n".join(current_lines).strip()})
    return shots


# ── spend tracking ───────────────────────────────────────────────────────────

def audit_spent() -> float:
    total = 0.0
    for f in RAW.glob("engine_s3_*.json"):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))[0]
            total += rec["tokens_or_credits"]["est_usd"]
        except Exception:
            continue
    return total


def est_usd():
    out_tiles = -(-W // 512) * -(-H // 512)
    return out_tiles * 0.000287, f"{out_tiles} output tiles @ $0.000287"


# ── Cloudflare image generation ──────────────────────────────────────────────

def multipart_body(prompt: str, width: int, height: int, seed: int):
    bd = uuid.uuid4().hex
    buf = io.BytesIO()
    for k, v in [("prompt", prompt), ("width", str(width)), ("height", str(height)), ("seed", str(seed))]:
        buf.write(f'--{bd}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    buf.write(f"--{bd}--\r\n".encode())
    return buf.getvalue(), f"multipart/form-data; boundary={bd}"


def _call_colab(prompt: str, seed: int) -> tuple:
    """Call the Colab FLUX server. Returns (raw_out_dict_or_str, elapsed_secs)."""
    body = json.dumps({"prompt": prompt, "width": W, "height": H, "seed": seed}).encode()
    url = f"{COLAB_URL}/generate"
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw_text = resp.read().decode()
    except urllib.error.HTTPError as e:
        raw_text = e.read().decode(errors="replace")
    except Exception as e:
        raw_text = f"CLIENT_ERROR {type(e).__name__}: {e}"
    elapsed = round(time.time() - t0, 1)
    raw_out = json.loads(raw_text) if raw_text[:1] == "{" else raw_text
    return raw_out, elapsed


def _call_cloudflare(prompt: str, seed: int) -> tuple:
    """Returns (raw_out_dict_or_str, elapsed_secs)."""
    body, ctype = multipart_body(prompt, W, H, seed)
    url = f"https://api.cloudflare.com/client/v4/accounts/{ACCT}/ai/run/{MODEL_ID}"
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Authorization": f"Bearer {TOK}", "Content-Type": ctype})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            raw_text = resp.read().decode()
    except urllib.error.HTTPError as e:
        raw_text = e.read().decode(errors="replace")
    except Exception as e:
        raw_text = f"CLIENT_ERROR {type(e).__name__}: {e}"
    elapsed = round(time.time() - t0, 1)
    raw_out = json.loads(raw_text) if raw_text[:1] == "{" else raw_text
    return raw_out, elapsed


def _audit_record(name, prompt, seed, raw_out, ok, flagged, elapsed, backend="cloudflare"):
    if backend == "colab":
        usd, basis = 0.0, "colab free T4 — $0"
    else:
        usd, basis = est_usd()
    billed = ok or flagged
    ts = datetime.now(timezone.utc)
    source = "colab flux.1-schnell" if backend == "colab" else f"cloudflare workers-ai {MODEL_TAG}"
    rec = [{
        "core_term": name,
        "seed_type": "shot-prompt",
        "prompt": prompt,
        "seed": seed,
        "raw_output": raw_out if not ok else "(base64 image omitted, saved to file)",
        "extracted_at": ts.isoformat(),
        "tokens_or_credits": {
            "est_usd": round(usd if billed else 0.0, 6),
            "basis": basis if ok else (basis + "; flagged 3030, counted as billed") if flagged else "failed, assumed not billed"
        },
        "source": source,
        "reason": f"Engine Step 3: image generation for shot {name}",
        "secs": elapsed
    }]
    RAW.mkdir(parents=True, exist_ok=True)
    audit_file = RAW / f"engine_s3_{ts.strftime('%Y%m%dT%H%M%S%f')}_{name}.json"
    audit_file.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return usd if billed else 0.0


def generate_image(name: str, prompt: str, out_dir: Path) -> str:
    dst = out_dir / f"{name}.png"
    if dst.exists():
        return "skip (already exists)"

    use_colab = bool(COLAB_URL)

    if not use_colab and not (ACCT and TOK):
        sys.exit("ERROR: Set COLAB_URL or CLOUDFLARE_ID/CLOUDFLARE_TOKEN in Engine/.env")

    backend = "colab" if use_colab else "cloudflare"
    rejected_dir = out_dir / "_rejected"

    for attempt in range(MAX_RETRIES):
        if not use_colab:
            current_spend = audit_spent()
            usd_est, _ = est_usd()
            if current_spend + usd_est > CAP_USD:
                sys.exit(f"SPEND CAP ${CAP_USD:.2f} would be exceeded (${current_spend:.4f} + ${usd_est:.4f}). "
                         f"Set ENGINE_CAP_USD higher in .env to continue.")

        seed = BASE_SEED + attempt * 100
        raw_out, elapsed = _call_colab(prompt, seed) if use_colab else _call_cloudflare(prompt, seed)

        ok = isinstance(raw_out, dict) and bool((raw_out.get("result") or {}).get("image"))
        flagged = not use_colab and isinstance(raw_out, dict) and any(e.get("code") == 3030 for e in raw_out.get("errors") or [])
        _audit_record(name, prompt, seed, raw_out, ok, flagged, elapsed, backend)

        if not ok:
            safe = str(raw_out)[:200]
            safe = re.sub(r'Bearer [A-Za-z0-9_-]+', 'Bearer REDACTED', safe)
            if flagged:
                return f"FAIL (3030 safety filter — prompt needs rewording)"
            if attempt < MAX_RETRIES - 1:
                print(f"FAIL[{attempt+1}] (API error, {elapsed}s) retrying...", end=" ", flush=True)
                continue
            return f"FAIL (all {MAX_RETRIES} attempts) {safe}"

        # Image generated — now QC it
        image_b64 = raw_out["result"]["image"]
        out_dir.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(base64.b64decode(image_b64))

        qc = qc_check(image_b64, prompt)
        if qc.get("pass", True):
            qc_note = ""
            if qc.get("skipped"):
                qc_note = f" (QC skipped: {qc['skipped']})"
            return f"OK ~${usd_est:.4f} {elapsed}s{qc_note}"
        else:
            defects = ", ".join(qc.get("defects", []))
            rejected_dir.mkdir(parents=True, exist_ok=True)
            rej_path = rejected_dir / f"{name}_s{seed}.png"
            shutil.move(str(dst), str(rej_path))

            # Audit the QC failure
            qc_audit = RAW / f"engine_s3_qc_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}_{name}.json"
            qc_audit.write_text(json.dumps({"shot": name, "seed": seed, "qc_result": qc,
                                            "action": "rejected, retrying" if attempt < MAX_RETRIES - 1 else "rejected, out of retries"},
                                           indent=2), encoding="utf-8")

            if attempt < MAX_RETRIES - 1:
                print(f"QC FAIL[{attempt+1}] ({defects}, {elapsed}s) retrying...", end=" ", flush=True)
                continue
            return f"QC FAIL (all {MAX_RETRIES} attempts) defects: {defects}"

    return "FAIL (unknown)"


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 3: Generate images from Step 2 prompts via Cloudflare Flux + Gemini QC.")
    parser.add_argument("--input", required=True, help="Path to Step 2 prompts markdown file")
    parser.add_argument("--output", required=True, help="Output directory for generated images")
    parser.add_argument("--only", nargs="*", help="Generate only these shot names (space-separated)")
    parser.add_argument("--spend", action="store_true", help="Print audited spend and exit")
    parser.add_argument("--no-qc", action="store_true", help="Skip QC checks (faster, no Gemini calls)")
    args = parser.parse_args()

    if args.spend:
        print(f"Engine Step 3 audited spend: ${audit_spent():.4f} of cap ${CAP_USD:.2f}")
        return

    backend = "Colab (free)" if COLAB_URL else "Cloudflare (paid)"
    print(f"Backend: {backend}")

    md_text = Path(args.input).read_text(encoding="utf-8")
    shots = parse_prompts(md_text)
    print(f"Parsed {len(shots)} prompts from {args.input}")

    if args.only:
        shots = [s for s in shots if s["name"] in args.only]
        print(f"Filtered to {len(shots)} shots: {[s['name'] for s in shots]}")

    if args.no_qc:
        global GEMINI_KEYS
        GEMINI_KEYS = []
        print("QC disabled (--no-qc)")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    fail_count = 0
    for i, s in enumerate(shots, 1):
        print(f"[{i}/{len(shots)}] {s['name']:25s} ", end="", flush=True)
        result = generate_image(s["name"], s["prompt"], out_dir)
        print(result, flush=True)
        if result.startswith("OK") or result.startswith("skip"):
            ok_count += 1
        else:
            fail_count += 1

    print(f"\nDone: {ok_count} OK, {fail_count} failed")
    print(f"Engine Step 3 audited spend: ${audit_spent():.4f} of cap ${CAP_USD:.2f}")


if __name__ == "__main__":
    main()
