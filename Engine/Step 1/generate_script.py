"""
Script Generation Engine — Step 1
Takes raw chapter text (.txt or .docx) and uses Gemini AI to produce a
structured beat-by-beat animation script.

Usage:
    python "Engine/Step 1/generate_script.py" --input "Engine/Chapter 1.docx" --output "Engine/Step 1/output/chapter1_script.txt"
    python "Engine/Step 1/generate_script.py" --input "Engine/Chapter 1.docx" --output "Engine/Step 1/output/chapter1_script.json" --format json
"""

import argparse, json, re, sys, os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# ── load .env safely (keys never printed/logged) ─────────────────────────────

def _load_env():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if load_dotenv and env_path.exists():
        load_dotenv(env_path)

_load_env()

_raw_keys = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
GEMINI_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]

SYSTEM_PROMPT = """\
You are a script writer for 2D animated educational videos aimed at Indian school children (Classes 1–9, CBSE English).

You will receive raw chapter content from a textbook. Your job is to turn it into a beat-by-beat animation script.

OUTPUT FORMAT — one line per beat, using this exact format:
shot_name | speaker | caption text

Rules:
- shot_name: a short snake_case name describing what is on screen (e.g. h_mom_close, s_adi_wave, i_bed, i_lamp_on)
  - Prefix h_ = character half-body shot, s_ = character scene/action shot, i_ = item/object close-up
- speaker: one of teacher, adi, mom, dad, narrator
  - teacher is the main on-screen instructor who opens and drives every lesson
  - mom and dad appear only inside story scenes and dialogues, never as the lesson host
  - narrator is used only for section titles and transitions
- caption text: what is spoken. Wrap key vocabulary words in *asterisks*.
- Use "== Section Title" lines to mark section breaks (intro, story, listen and repeat, quiz, word cards, activities, outro).
- The teacher opens the lesson, welcomes the children, and previews the topic.
- The story section should be vivid, simple, and describe actions Adi does on screen.
- Include "listen and repeat" sections where the child repeats after the speaker.
- Quiz questions should be answerable out loud by the child.
- Word cards: list each vocabulary word with a pronunciation split (e.g. Ta-ble) and a tip.
- End with an outro (teacher closes).
- Keep language simple. Short sentences. Target ~20 minutes of narration when read at a slow, clear pace.
- The teacher speaks most lines. Adi speaks his own dialogue in quotes.

CRITICAL — shot-to-text coherence:
- The shot_name describes what is visually on screen. The spoken text MUST reference what is visible in that shot.
- If the shot shows a character interacting with an object (e.g. s_adi_point_bed), the spoken text must mention that object explicitly. WRONG: "He points his finger." RIGHT: "He points his finger towards his bed."
- If the shot is an item close-up (e.g. i_lamp_on), the text must talk about that item.
- Never leave the connection between the visual and the dialogue vague or implicit. The child watching must hear exactly what they see.
- Add flags after the text when needed:
  - walk = character walks on screen
  - alt=other_shot = flip between two shots (for jumping etc.)
  - card=Word;split=Wo-rd = show a word card
  - rate=-20% = slower speech

Output ONLY the script lines. No commentary, no markdown fences, no explanation."""

# ── read input (txt or docx) ─────────────────────────────────────────────────

def read_input(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return path.read_text(encoding="utf-8")


# ── AI call via Gemini ────────────────────────────────────────────────────────

def call_ai(raw_chapter: str) -> str:
    import urllib.request, urllib.error, time

    if not GEMINI_KEYS:
        print("ERROR: No GEMINI_API_KEY found in Engine/.env", file=sys.stderr)
        sys.exit(1)

    body = json.dumps({
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"Here is the raw chapter content. Turn it into an animation script:\n\n{raw_chapter}"}],
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 8000,
        },
    }).encode()

    for i, key in enumerate(GEMINI_KEYS):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                print(f"  Success with key #{i+1}")
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode() if e.fp else ""
            if e.code == 429 and i < len(GEMINI_KEYS) - 1:
                print(f"  Key #{i+1} rate limited, trying next key...")
                continue
            safe_err = re.sub(r'key=[A-Za-z0-9_-]+', 'key=REDACTED', err_body)
            print(f"ERROR: Gemini API returned {e.code} on key #{i+1}", file=sys.stderr)
            print(f"  Detail: {safe_err[:500]}", file=sys.stderr)
            if i < len(GEMINI_KEYS) - 1:
                print(f"  Trying next key...", file=sys.stderr)
                continue
            sys.exit(1)
        except urllib.error.URLError as e:
            print(f"ERROR: Cannot reach Gemini API: {e.reason}", file=sys.stderr)
            sys.exit(1)

    print("ERROR: All keys exhausted", file=sys.stderr)
    sys.exit(1)


# ── parse AI output back into structured beats ───────────────────────────────

SPEAKERS = {"mom", "adi", "dad", "sam", "narrator", "teacher"}

def parse_script_output(ai_text: str) -> list[dict]:
    beats = []
    for line in ai_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("=="):
            beats.append({"type": "section", "heading": line.lstrip("= ").strip()})
            continue

        if line.startswith("#"):
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue

        shot = parts[0]
        speaker = parts[1].lower()
        text = "|".join(parts[2:]).strip()
        focus = re.findall(r"\*(\w+)\*", text)

        if speaker not in SPEAKERS:
            speaker = "mom"

        beats.append({
            "shot": shot,
            "speaker": speaker,
            "text": text,
            "focus_words": focus,
        })

    return beats


# ── output formatters ─────────────────────────────────────────────────────────

def beats_to_txt(beats: list[dict]) -> str:
    lines = []
    for b in beats:
        if b.get("type") == "section":
            lines.append(f"\n== {b['heading']}")
            continue
        focus = ""
        if b.get("focus_words"):
            focus = "  # focus: " + ", ".join(b["focus_words"])
        lines.append(f'{b["shot"]:<16} | {b["speaker"]:<8} | {b["text"]}{focus}')
    return "\n".join(lines).strip() + "\n"


def beats_to_json(beats: list[dict]) -> str:
    return json.dumps(beats, indent=2, ensure_ascii=False)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 1: AI-powered script generation from raw chapter text.")
    parser.add_argument("--input", required=True, help="Path to raw chapter file (.txt or .docx)")
    parser.add_argument("--output", required=True, help="Path for the generated script")
    parser.add_argument("--format", choices=["txt", "json"], default="txt", help="Output format (default: txt)")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"ERROR: Input file not found: {inp}", file=sys.stderr)
        sys.exit(1)

    raw = read_input(inp)
    print(f"Read {len(raw)} chars from {args.input}")
    print("Sending to Gemini...")

    ai_output = call_ai(raw)
    print(f"AI returned {len(ai_output)} chars")

    beats = parse_script_output(ai_output)

    if args.format == "json":
        out = beats_to_json(beats)
    else:
        out = beats_to_txt(beats)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(out, encoding="utf-8")
    print(f"Done — {len(beats)} beats written to {args.output}")


if __name__ == "__main__":
    main()
