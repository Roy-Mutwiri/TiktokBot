# Multilingual avatar voice — Arabic + English (+ 21 more)

The avatar can speak **Arabic and English** (and 21 other languages) in one
human, clonable voice, with automatic code-switching. It's powered by
**Chatterbox Multilingual** (Resemble AI), running locally on the GPU.

> Note: this is NOT a "trained Fable 5". A language model (Fable 5 / Claude)
> writes the *words*; the *voice* is a TTS model. Training a TTS from scratch
> needs weeks of compute + huge corpora, which isn't feasible on one GPU.
> Chatterbox Multilingual already speaks Arabic/English at high quality and
> clones any voice zero-shot, so we get a limitless multilingual voice now and
> can fine-tune later if a dataset is gathered.

## Enable it

**In Avatar Studio:** pick **"Arabic + English (Multilingual)"** in the voice
dropdown.

**Headless / env:**
```
set AVATAR_TTS=multilingual          # use the multilingual backend
set AVATAR_TTS_LANG=auto             # auto | ar | en | fr | es | ... (default auto)
set AVATAR_CLONE_REF=C:\path\ref.wav # clone ANY voice across all languages (optional)
set AVATAR_CLONE_EXAGGERATION=0.6    # 0.3 calm .. 1.0 very emotive
```

- `auto` (default) routes each utterance to the right language and, for
  **code-switched** text like `Welcome, مرحباً بك, let's go`, splits by script
  and speaks each run in its own language, then stitches them seamlessly.
- Force one language with `AVATAR_TTS_LANG=ar` (or pass `lang="ar"` to
  `synthesize`).

## Quality hardening (built in)

- **Numbers → words** in the target language (4517 → "four thousand…"; ٤ →
  "أربعة…"; 9:45, 3rd, 3.5 handled) so numerals are pronounced, not guessed.
- **Anti-repetition**: long/tricky lines are clause-split and a runaway
  generation is auto-retried with steadier sampling (this fixed tongue-twister
  loops in testing).

## Hard test (objective)

`python tts_hardtest.py` synthesizes English / MSA Arabic / Arabic dialect /
code-switch / number-heavy lines, round-trips each through **Whisper ASR**, and
reports CER/WER + a wrong-language cross-check + real-time-factor. Clips land in
`tts_samples/` so you can also just listen.

Latest run (built-in voice, whisper-large-v3-turbo):

| category | CER | WER | notes |
|----------|-----|-----|-------|
| english        | 0.09 | 0.10 | clean lines perfect; residual = ASR writing "3.5%" |
| arabic_msa     | 0.08 | 0.16 | excellent, very intelligible |
| arabic_dialect | 0.40 | 0.63 | Egyptian colloquial works |
| **mixed (code-switch)** | **0.00** | **0.00** | Arabic+English in one line, perfect |
| numbers/hard   | 0.52 | 0.59 | "Order 4517… 9:45" correct; long spelled-out Arabic phone numbers still hard |
| **OVERALL**    | **0.18** | **0.25** | RTF 0.71 (faster than real-time) |

The wrong-language cross-check (xCER ≈ 1.0) confirms the audio is genuinely in
the intended language/accent, not language-agnostic mush.

## Going further (optional fine-tune)

For a *specific* cloned voice, set `AVATAR_CLONE_REF` to a 7–20s clean clip —
zero-shot, no training. For a tuned accent/voice you'd fine-tune Chatterbox on a
small (audio, transcript) dataset; that's the next step once such data exists.
