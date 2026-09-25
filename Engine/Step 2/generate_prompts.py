"""
Image Prompt Generation Engine — Step 2
Reads a script scene and generates image-generation prompts for every unique shot.
Each prompt produces one 16:9 desktop frame.

Usage:
    python "Engine/Step 2/generate_prompts.py" --input "Engine/Step 2/input/scene.txt" --output "Engine/Step 2/output/scene_prompts.md"
"""

import argparse, json, re, sys, os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

def _load_env():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if load_dotenv and env_path.exists():
        load_dotenv(env_path)

_load_env()

_raw_keys = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
GEMINI_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]

CHARACTER_DEFS = {
    "teacher": (
        "A stylized 3D animated adult woman character, warm brown skin, wearing a royal-blue "
        "silk saree with gold border, gold-rimmed glasses, hair in a neat low bun with jasmine "
        "flowers, kind smile, elegant posture"
    ),
    "adi": (
        "A stylized 3D animated small cartoon character with oversized round head, "
        "warm brown skin, wearing a mustard-yellow t-shirt, turned-up denim shorts, "
        "white sneakers, messy black hair, big round cartoon eyes, happy smile"
    ),
    "mom": (
        "A stylized 3D animated adult woman character, warm brown skin, wearing a teal "
        "churidar with coral dupatta, black hair in a loose braid, gentle caring expression"
    ),
    "dad": (
        "A stylized 3D animated adult man character, warm brown skin, wearing a light-blue "
        "collared shirt and charcoal trousers, glasses, neat black hair, friendly smile"
    ),
    "sam": (
        "A stylized 3D animated small cartoon character with oversized round head, "
        "warm brown skin, wearing a green polo t-shirt and khaki shorts, neat black hair "
        "parted to the side, friendly curious expression"
    ),
}

STYLE_LOCK = (
    "3D Pixar-style render, soft ambient lighting, vibrant saturated colors, "
    "smooth skin with subtle subsurface scattering, large expressive eyes, "
    "rounded friendly features, clean sharp details, no text or watermarks, "
    "16:9 landscape desktop aspect ratio, warm color palette"
)

SYSTEM_PROMPT = f"""\
You are an image prompt writer for an animated educational video engine.

You will receive a list of unique shot names with context. Write ONE image-generation prompt per shot.

CHARACTER DEFINITIONS (use exactly, never change appearance):
- Teacher: {CHARACTER_DEFS["teacher"]}
- Adi: {CHARACTER_DEFS["adi"]}
- Mom: {CHARACTER_DEFS["mom"]}
- Dad: {CHARACTER_DEFS["dad"]}
- Sam: {CHARACTER_DEFS["sam"]}

STYLE (append to every prompt): {STYLE_LOCK}

RULES:
1. Each prompt = ONE frame. Describe what is visible.
2. Format: 16:9 landscape (desktop). Never portrait.
3. Shot prefixes define framing:
   - h_ = half-body (waist up, character centered)
   - s_ = scene (full body in environment, showing the action)
   - i_ = item close-up (object fills frame, no characters visible unless hands needed)
4. Adi's bedroom LAYOUT (use this spatial arrangement consistently in every room shot):
   - LEFT WALL: blue bed against the left wall, neatly made with blue blanket and pillow.
   - RIGHT SIDE, near window: wooden study table with ONE plain white desk lamp (ordinary lamp with shade, no face, no eyes, not anthropomorphic) and a small green wooden chair tucked in front of the table.
   - FAR RIGHT CORNER: big closed wooden toy box (chest with a lid, not an open crate) with colorful toys visible only when the lid is open.
   - BACK WALL: window with cream curtains, centered.
   - FLOOR: light wooden floor with ONE soft rug in the center. Floor is CLEAN — no toys, no blocks, no clutter anywhere on the floor.
   - WALLS: pale warm-yellow.
   - There is exactly ONE lamp in the room, on the study table. Never two lamps.
   - The open space between bed (left) and table (right) is where Adi walks and stands. Keep this path clear.
5. Teacher shots: bright colorful classroom, green chalkboard behind her, educational posters.
6. Characters must look IDENTICAL across all prompts.
7. Describe exact pose, expression, gesture, and visible objects.
8. If the shot shows interaction with an object, the character must be physically touching/pointing at/holding it.
9. Eyes always open. Adi always happy, eyes on the action.
10. Objects are NEVER anthropomorphic — no faces, no eyes, no expressions on furniture, lamps, toys, or any inanimate object. Only characters have faces.
11. The lamp is an ordinary plain white desk lamp with a simple shade. NOT a nightlight, NOT a blob, NOT a character. Just a normal lamp.
12. No text, subtitles, or watermarks in the image.
13. Never use the word "child", "kid", "boy", "girl", or any age number in prompts. Use only the character definitions above which say "cartoon character".

OUTPUT FORMAT — for each shot output exactly:
### shot_name
[full prompt text including style lock]

Nothing else. No commentary."""


def extract_shots(script_text: str) -> list[dict]:
    shots_seen = set()
    shots = []
    for line in script_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("==") or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        shot = parts[0]
        speaker = parts[1]
        text = parts[2]
        if shot == "narrator":
            continue
        if shot not in shots_seen:
            shots_seen.add(shot)
            shots.append({"shot": shot, "speaker": speaker, "text": text})
    return shots


def call_ai(prompt_text: str) -> str:
    import urllib.request, urllib.error

    if not GEMINI_KEYS:
        print("ERROR: No GEMINI_API_KEY found in Engine/.env", file=sys.stderr)
        sys.exit(1)

    body = json.dumps({
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 16000},
    }).encode()

    for i, key in enumerate(GEMINI_KEYS):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
                print(f"  Success with key #{i+1}")
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode() if e.fp else ""
            if e.code == 429 and i < len(GEMINI_KEYS) - 1:
                print(f"  Key #{i+1} rate limited, trying next...")
                continue
            safe_err = re.sub(r'key=[A-Za-z0-9_-]+', 'key=REDACTED', err_body)
            print(f"ERROR: Gemini returned {e.code}", file=sys.stderr)
            print(f"  {safe_err[:500]}", file=sys.stderr)
            if i < len(GEMINI_KEYS) - 1:
                continue
            sys.exit(1)
        except urllib.error.URLError as e:
            print(f"ERROR: Cannot reach Gemini: {e.reason}", file=sys.stderr)
            sys.exit(1)

    print("ERROR: All keys exhausted", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Step 2: Generate image prompts from script scene.")
    parser.add_argument("--input", required=True, help="Path to scene script file")
    parser.add_argument("--output", required=True, help="Path for generated prompts")
    args = parser.parse_args()

    script_text = Path(args.input).read_text(encoding="utf-8")
    shots = extract_shots(script_text)
    print(f"Found {len(shots)} unique shots")

    shot_list = "\n".join(
        f"- {s['shot']} (speaker: {s['speaker']}) — context: \"{s['text']}\""
        for s in shots
    )
    user_prompt = f"Generate one image prompt per shot:\n\n{shot_list}"

    print("Sending to Gemini...")
    ai_output = call_ai(user_prompt)
    print(f"AI returned {len(ai_output)} chars")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(ai_output, encoding="utf-8")
    print(f"Done — {len(shots)} prompts written to {args.output}")


if __name__ == "__main__":
    main()
