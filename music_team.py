"""
rin.music — 音楽制作チームエージェント

メンバー:
  👔 yasu         : CEO — 戦略・指揮
  🎸 蒼牙（そうが）: サウンドディレクター — 全体ビジョン・ジャンル設計
  🎹 澪（みお）   : コード進行エージェント
  🥁 颯（はやて） : アレンジメントエージェント
  🎚️  凛（りん）   : サウンドデザイン & プロダクションノート

使い方:
  python music_team.py       チャットモードで起動
  「おつ～」                  会話を保存して終了
  「よろ～」                  前回の会話を読み込んで続きから
"""

import os
import re
import json
import datetime
import anthropic
from pathlib import Path

MODEL  = "claude-opus-4-6"
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SUPPORTED_LANGUAGES = {"ja": "日本語", "en": "English"}

SAVE_KEYWORD = "おつ～"
LOAD_KEYWORD = "よろ～"
SESSION_DIR  = Path("sessions")
MUSIC_SESSION_FILE = SESSION_DIR / "music_latest.json"


# ---------------------------------------------------------------------------
# セッション管理
# ---------------------------------------------------------------------------

def save_music_session(messages: list) -> str:
    SESSION_DIR.mkdir(exist_ok=True)
    data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "messages":  messages,
    }
    MUSIC_SESSION_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(MUSIC_SESSION_FILE)


def load_music_session() -> list:
    if not MUSIC_SESSION_FILE.exists():
        return []
    data = json.loads(MUSIC_SESSION_FILE.read_text(encoding="utf-8"))
    return data.get("messages", [])


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _stream_response(system: str, messages: list, max_tokens: int = 4096) -> str:
    """ストリーミングでAPIを呼び出し、完全なテキストを返す。"""
    with client.messages.stream(
        model=MODEL,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=system,
        messages=messages,
    ) as stream:
        chunks = []
        for event in stream:
            if (
                event.type == "content_block_delta"
                and event.delta.type == "text_delta"
            ):
                print(event.delta.text, end="", flush=True)
                chunks.append(event.delta.text)
        print()
        return "".join(chunks)


def _safe_filename(text: str, max_len: int = 40) -> str:
    cleaned = re.sub(r'[^\w\s\-]', '', text, flags=re.UNICODE)
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    return cleaned[:max_len] if cleaned else "music"


def _parse_field(text: str, key: str, default: str = "") -> str:
    m = re.search(rf'\*\*{re.escape(key)}\*\*\s*[:\uff1a]\s*(.+)', text)
    return m.group(1).strip() if m else default


def _extract_directive(mandate: str, agent_name: str) -> str:
    pattern = rf'### Directive[^#]*?{re.escape(agent_name)}.*?\n(.*?)(?=###|\Z)'
    m = re.search(pattern, mandate, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# yasu — チャットシステムプロンプト（音楽制作特化）
# ---------------------------------------------------------------------------

YASU_MUSIC_CHAT_SYSTEM = """\
You are yasu, the CEO of rin.music, a creative music production company.
You're having a natural, casual conversation with your client about the music they want to make.
You focus on the MUSICAL side: genre, mood, chord feel, rhythm, arrangement, instrumentation.

Your team:
- 蒼牙 (Souga): Sound Director — overall musical vision, genre design, sonic identity
- 澪 (Mio): Chord/Harmony Agent — chord progressions, key, harmonic color
- 颯 (Hayate): Arrangement Agent — rhythm, tempo, song structure, instrumentation
- 凛 (Rin): Sound Design & Production Notes — final production advice

Personality: creative, enthusiastic, concise. You speak casually but with professional instincts.
Always reply in the same language the user uses (Japanese or English).

When the user wants music ACTUALLY CREATED (chord progressions, arrangement, production guide — not just discussed),
output this line at the very end of your response — nothing after it:
ACTION: CREATE_MUSIC | <one-line summary of what to create>

Only add the ACTION line when you are ready to start production.
Do NOT add it for conceptual discussions or questions.
"""


def _music_chat_turn(messages: list) -> str:
    """yasuとの音楽制作チャットターン。"""
    with client.messages.stream(
        model=MODEL,
        max_tokens=1024,
        system=YASU_MUSIC_CHAT_SYSTEM,
        messages=messages,
    ) as stream:
        chunks = []
        for event in stream:
            if (
                event.type == "content_block_delta"
                and event.delta.type == "text_delta"
            ):
                print(event.delta.text, end="", flush=True)
                chunks.append(event.delta.text)
        print()
        return "".join(chunks)


def _parse_action(response: str) -> str | None:
    """ACTION: CREATE_MUSIC | <request> を抽出する。なければ None。"""
    m = re.search(r'ACTION:\s*CREATE_MUSIC\s*\|\s*(.+)', response)
    return m.group(1).strip() if m else None


def _visible_response(response: str) -> str:
    """ACTION行をユーザーに見せないよう除去したテキスト。"""
    return re.sub(r'\nACTION:\s*CREATE_MUSIC\s*\|.+', '', response).rstrip()


# ---------------------------------------------------------------------------
# 音楽制作パイプライン（CEO→チーム）
# ---------------------------------------------------------------------------

MUSIC_CEO_SYSTEM = """\
You are yasu, the CEO of rin.music, a creative music production company.
Your job is to receive a music production request and write a strategic production mandate
that directs your team (Sound Director, Chord Agent, Arrangement Agent, Producer).

Analyze the request and decide:
- Genre (e.g. "J-POP", "City Pop", "Lo-fi Hip-hop", "EDM", "Jazz-Fusion", "Indie Folk")
- Language ("ja" for Japanese context, "en" for English context) — default "ja" if ambiguous
- Title Concept (working title)
- Key (e.g. "C major", "A minor", "F# minor")
- BPM Range (e.g. "80–90", "120–128")
- Mood (e.g. "melancholic but hopeful", "energetic and euphoric")
- Tone and style

Output your mandate in the following exact format (fill in every field):

**Genre:** <genre>
**Language:** <ja or en>
**Title Concept:** <working title>
**Key:** <key>
**BPM Range:** <bpm range>
**Mood:** <mood>
**Tone:** <tone and style>

### Directive — Sound Director
<instructions for ソウルスミス>

### Directive — Chord Agent
<instructions for ハーモニー>

### Directive — Arrangement Agent
<instructions for リズム>

### Directive — Producer
<instructions for プロデューサー>

### CEO Note
<any additional strategic notes or creative vision>
"""


def _sound_director_system(genre: str, language: str, key: str, bpm: str, directive: str) -> str:
    if language == "en":
        return f"""\
You are 蒼牙 (Souga), a visionary Sound Director specializing in {genre}.
Key: {key} | BPM: {bpm}
CEO directive: {directive}

Define the overall sonic identity and musical vision for this track.

Output format:
## Sonic Identity
## Reference Sounds & Artists
## Instrumentation Palette
## Key Emotional Moments
## Production Style Notes
"""
    return f"""\
あなたは蒼牙（そうが）、{genre}専門のビジョナリー・サウンドディレクターです。
キー: {key} | BPM: {bpm}
CEOからの指示: {directive}

このトラックの全体的なサウンドアイデンティティと音楽的ビジョンを定義してください。

出力形式:
## サウンドアイデンティティ
## 参照サウンド・アーティスト
## 使用楽器パレット
## 重要な感情的瞬間
## プロダクションスタイル
"""


def _chord_agent_system(genre: str, language: str, key: str, directive: str) -> str:
    if language == "en":
        return f"""\
You are 澪 (Mio), a harmony and chord specialist for {genre}.
Key: {key}
CEO directive: {directive}

Design detailed chord progressions for each section of the song.
Use standard chord notation (e.g. Cmaj7, Am7, F, G/B) and Roman numerals.

Output format:
## Key & Scale
## Intro Chords
## Verse (Aメロ) Chords
## Pre-Chorus (Bメロ) Chords
## Chorus (サビ) Chords
## Bridge Chords (if applicable)
## Outro Chords
## Harmonic Color Notes
"""
    return f"""\
あなたは澪（みお）、{genre}専門のハーモニー＆コードスペシャリストです。
キー: {key}
CEOからの指示: {directive}

楽曲の各セクションに対して詳細なコード進行を設計してください。
コードネーム（例: Cmaj7, Am7, F, G/B）とローマ数字表記を使用してください。

出力形式:
## キー & スケール
## イントロのコード
## Aメロのコード
## Bメロのコード
## サビのコード
## ブリッジのコード（該当する場合）
## アウトロのコード
## ハーモニーカラーのメモ
"""


def _arrangement_agent_system(genre: str, language: str, bpm: str, directive: str) -> str:
    if language == "en":
        return f"""\
You are 颯 (Hayate), an arrangement specialist for {genre}.
BPM: {bpm}
CEO directive: {directive}

Design the complete song arrangement: structure, rhythm patterns, and instrumentation layering.

Output format:
## Song Structure (timeline with bars/sections)
## Rhythm & Groove Pattern
## Drum/Percussion Design
## Bass Pattern
## Instrument Layering Guide
## Dynamics & Energy Flow
## Transition & Fill Ideas
"""
    return f"""\
あなたは颯（はやて）、{genre}専門のアレンジメントスペシャリストです。
BPM: {bpm}
CEOからの指示: {directive}

楽曲の完全なアレンジメントを設計してください: 構成、リズムパターン、楽器のレイヤリング。

出力形式:
## 楽曲構成（タイムライン・小節数）
## リズム＆グルーヴパターン
## ドラム・パーカッションデザイン
## ベースパターン
## 楽器レイヤリングガイド
## ダイナミクス＆エネルギーフロー
## トランジション＆フィルのアイデア
"""


def _producer_system(genre: str, language: str, directive: str) -> str:
    if language == "en":
        return f"""\
You are 凛 (Rin), a veteran music producer specializing in {genre}.
CEO directive: {directive}

Review the sound vision, chords, and arrangement provided, then write a complete production guide.

Output format:
## Overall Production Assessment
## Sound Design Recommendations
## Mix Balance Guide (levels, panning, EQ focus)
## Effects & Processing (reverb, delay, compression)
## DAW / Plugin Recommendations
## Final Production Notes
"""
    return f"""\
あなたは凛（りん）、{genre}に精通したベテラン音楽プロデューサーです。
CEOからの指示: {directive}

提供されたサウンドビジョン・コード・アレンジメントをレビューし、完全なプロダクションガイドを作成してください。

出力形式:
## 総合プロダクション評価
## サウンドデザイン推奨事項
## ミックスバランスガイド（レベル・パンニング・EQフォーカス）
## エフェクト＆プロセッシング（リバーブ・ディレイ・コンプレッション）
## DAW・プラグイン推奨
## 最終プロダクションノート
"""


def save_music_file(result: dict, request: str) -> str:
    output_dir = Path("music")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _safe_filename(request)
    filename = output_dir / f"{timestamp}_{slug}.md"
    lang_label = SUPPORTED_LANGUAGES.get(result.get("language", "ja"), "日本語")
    content = f"""\
# 音楽制作ガイド / Music Production Guide

**リクエスト:** {request}
**ジャンル:** {result.get('genre', '')}
**キー:** {result.get('key', '')}
**BPM:** {result.get('bpm', '')}
**言語:** {lang_label}
**作成日時:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## CEO 制作指令書

{result['mandate']}

---

## サウンドディレクション（ソウルスミス）

{result['sound_direction']}

---

## コード進行（ハーモニー）

{result['chords']}

---

## アレンジメント（リズム）

{result['arrangement']}

---

## プロダクションガイド（プロデューサー）

{result['production']}
"""
    filename.write_text(content, encoding="utf-8")
    return str(filename)


def create_music(user_request: str) -> dict:
    """CEO→チームエージェントの音楽制作パイプライン。"""
    print("\n👔 [yasu / CEO] 制作指令書を作成中...\n")
    mandate = _stream_response(
        MUSIC_CEO_SYSTEM,
        [{"role": "user", "content": user_request}],
    )

    genre    = _parse_field(mandate, "Genre",         default="J-POP")
    language = _parse_field(mandate, "Language",      default="ja").strip().lower()
    title    = _parse_field(mandate, "Title Concept", default="Untitled")
    key      = _parse_field(mandate, "Key",            default="C major")
    bpm      = _parse_field(mandate, "BPM Range",      default="120")
    if language not in SUPPORTED_LANGUAGES:
        language = "ja"

    lang_label = SUPPORTED_LANGUAGES[language]
    print(f"\n   📋 Genre: {genre} | Key: {key} | BPM: {bpm} | Language: {lang_label} | Title: {title}")

    dir_sound  = _extract_directive(mandate, "Sound Director")
    dir_chord  = _extract_directive(mandate, "Chord Agent")
    dir_arr    = _extract_directive(mandate, "Arrangement Agent")
    dir_prod   = _extract_directive(mandate, "Producer")

    base_prompt = (
        f"Genre: {genre}\nKey: {key}\nBPM: {bpm}\nRequest: {user_request}"
        if language == "en" else
        f"ジャンル: {genre}\nキー: {key}\nBPM: {bpm}\nリクエスト: {user_request}"
    )

    # ── ソウルスミス ──
    print(f"\n🎸 [蒼牙 / サウンドディレクター] サウンドビジョンを構築中...\n")
    sound_direction = _stream_response(
        _sound_director_system(genre, language, key, bpm, dir_sound),
        [{"role": "user", "content": base_prompt}],
    )

    # ── ハーモニー ──
    print(f"\n🎹 [澪 / コード進行エージェント] コード進行を設計中...\n")
    chord_prompt = (
        base_prompt + f"\n\nSound Vision:\n{sound_direction}"
        if language == "en" else
        base_prompt + f"\n\nサウンドビジョン:\n{sound_direction}"
    )
    chords = _stream_response(
        _chord_agent_system(genre, language, key, dir_chord),
        [{"role": "user", "content": chord_prompt}],
    )

    # ── リズム ──
    print(f"\n🥁 [颯 / アレンジメントエージェント] アレンジメントを構成中...\n")
    arr_prompt = (
        base_prompt + f"\n\nSound Vision:\n{sound_direction}\n\nChords:\n{chords}"
        if language == "en" else
        base_prompt + f"\n\nサウンドビジョン:\n{sound_direction}\n\nコード進行:\n{chords}"
    )
    arrangement = _stream_response(
        _arrangement_agent_system(genre, language, bpm, dir_arr),
        [{"role": "user", "content": arr_prompt}],
    )

    # ── プロデューサー ──
    print(f"\n🎚️  [凛 / サウンドデザイン] プロダクションガイドを作成中...\n")
    prod_prompt = (
        base_prompt
        + f"\n\nSound Vision:\n{sound_direction}"
        + f"\n\nChords:\n{chords}"
        + f"\n\nArrangement:\n{arrangement}"
        if language == "en" else
        base_prompt
        + f"\n\nサウンドビジョン:\n{sound_direction}"
        + f"\n\nコード進行:\n{chords}"
        + f"\n\nアレンジメント:\n{arrangement}"
    )
    production = _stream_response(
        _producer_system(genre, language, dir_prod),
        [{"role": "user", "content": prod_prompt}],
    )

    result = {
        "mandate": mandate, "genre": genre, "language": language,
        "title": title, "key": key, "bpm": bpm,
        "sound_direction": sound_direction, "chords": chords,
        "arrangement": arrangement, "production": production,
    }
    saved = save_music_file(result, user_request)
    result["saved_to"] = saved

    print(f"\n✅ 完成！  Saved → {saved}")
    return result


# ---------------------------------------------------------------------------
# チャット REPL
# ---------------------------------------------------------------------------

def run_music_chat() -> None:
    print("=" * 60)
    print("🎸 rin.music — 音楽制作チーム")
    print(f'   「{SAVE_KEYWORD}」で保存  「{LOAD_KEYWORD}」で続きから')
    print("=" * 60)

    messages: list = []

    # 開幕挨拶
    print("\n👔 yasu: ", end="", flush=True)
    greeting = _music_chat_turn([{"role": "user", "content": "音楽制作セッション開始。軽く挨拶して。"}])
    messages.append({"role": "user",      "content": "セッション開始"})
    messages.append({"role": "assistant", "content": greeting})

    while True:
        try:
            user_input = input("\nあなた: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not user_input:
            continue

        # ── 保存 ──
        if user_input == SAVE_KEYWORD:
            path = save_music_session(messages)
            print(f"\n💾 会話を保存しました → {path}")
            print("👔 yasu: またね！お疲れ～")
            break

        # ── 読み込み ──
        if user_input == LOAD_KEYWORD:
            loaded = load_music_session()
            if not loaded:
                print("📂 保存された会話が見つかりませんでした。")
                continue
            messages = loaded
            turns = sum(1 for m in messages if m["role"] == "user")
            print(f"\n📂 前回の会話を読み込みました（{turns}ターン）\n")
            print("👔 yasu: ", end="", flush=True)
            resume_resp = _music_chat_turn(
                messages + [{"role": "user", "content":
                    "前回の会話の続きです。前回どんな話をしていたか簡単に振り返ってから続けて。"}]
            )
            messages.append({"role": "user",      "content": LOAD_KEYWORD})
            messages.append({"role": "assistant", "content": resume_resp})
            continue

        # ── 通常会話 ──
        messages.append({"role": "user", "content": user_input})
        print("\n👔 yasu: ", end="", flush=True)
        response = _music_chat_turn(messages)

        action_request = _parse_action(response)
        visible = _visible_response(response)

        messages.append({"role": "assistant", "content": visible})

        # ── 音楽制作アクション ──
        if action_request:
            print("\n" + "─" * 60)
            result = create_music(action_request)
            print("─" * 60)
            summary = (
                f"[音楽制作完了] タイトル: {result['title']} / "
                f"ジャンル: {result['genre']} / キー: {result['key']} / "
                f"保存先: {result['saved_to']}"
            )
            messages.append({"role": "user",      "content": "[音楽制作完了の報告]"})
            messages.append({"role": "assistant", "content": summary})


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_music_chat()
