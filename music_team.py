"""
rin.music — 音楽制作チームエージェント

メンバー:
  👔 yasu       : CEO — 戦略・指揮
  🎼 ハル（はる）: 作曲エージェント — メロディ・コード進行
  🎹 ソラ        : アレンジエージェント — 編曲・楽器構成
  🎚️  カイ        : プロデュースエージェント — 制作指針・総括

使い方:
  python music_team.py       チャットモードで起動
  「おつ～」                   会話を保存して終了
  「よろ～」                   前回の会話を読み込んで続きから
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

SAVE_KEYWORD  = "おつ～"
LOAD_KEYWORD  = "よろ～"
SESSION_DIR   = Path("sessions")
SESSION_FILE  = SESSION_DIR / "music_latest.json"
OUTPUT_DIR    = Path("music")


# ---------------------------------------------------------------------------
# セッション管理
# ---------------------------------------------------------------------------

def save_session(messages: list) -> str:
    SESSION_DIR.mkdir(exist_ok=True)
    data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "messages":  messages,
    }
    SESSION_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(SESSION_FILE)


def load_session() -> list:
    if not SESSION_FILE.exists():
        return []
    data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
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
# yasu — チャットシステムプロンプト
# ---------------------------------------------------------------------------

YASU_MUSIC_CHAT_SYSTEM = """\
You are yasu, the CEO of rin.music, a creative music production company.
You're having a natural, casual conversation with your client about their music.

Your team:
- ハル（はる）: Composition Agent — creates melodies, chord progressions, motifs
- ソラ: Arrangement Agent — handles instrumentation, texture, and structure
- カイ: Production Agent — oversees production direction and final output

Personality: creative, enthusiastic, concise. You speak casually but with professional instincts.
Always reply in the same language the user uses (Japanese or English).

When the user wants music ACTUALLY CREATED (not just discussed), \
output this line at the very end of your response — nothing after it:
ACTION: CREATE_MUSIC | <one-line summary of what to create>

Only add the ACTION line when you are ready to start production. \
Do NOT add it for conceptual discussions or questions.
"""


def _chat_turn(messages: list) -> str:
    """yasuとのチャットターン（thinking なし・短めのトークン）。"""
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

CEO_MUSIC_SYSTEM = """\
You are yasu, the CEO of rin.music, a creative music production company.
Your job is to receive a user's music request and produce a strategic production mandate \
that directs your team of agents (Composition Agent, Arrangement Agent, Production Agent).

Analyze the request and decide:
- Genre (be specific, e.g. "J-POP Ballad", "City Pop", "Indie Folk", "Electronic", "Jazz")
- Language ("ja" for Japanese context, "en" for English context) — \
  infer from the request language; default to "ja" if ambiguous
- Title concept (a working title for the piece)
- Target audience
- Emotional goal
- Tempo and feel (e.g. "slow and melancholic", "upbeat and energetic")
- Key signature suggestion (e.g. "C major", "A minor")
- Specific directives for each team member

Output your mandate in the following exact format (fill in every field):

**Genre:** <genre>
**Language:** <ja or en>
**Title Concept:** <working title>
**Target Audience:** <audience>
**Emotional Goal:** <goal>
**Tempo and Feel:** <tempo and feel>
**Key Suggestion:** <key>

### Directive — Composition Agent
<instructions for the composition agent>

### Directive — Arrangement Agent
<instructions for the arrangement agent>

### Directive — Production Agent
<instructions for the production agent>

### CEO Note
<any additional strategic notes or creative vision>
"""


def _composition_system(genre: str, language: str, directive: str) -> str:
    if language == "en":
        return f"""\
You are ハル (Haru), a composer specializing in {genre}.
The CEO has given you the following directive:
{directive}

Develop the melodic and harmonic foundation for the piece.

Output format:
## Melody Concept
## Chord Progression
## Key Motifs & Hooks
## Rhythmic Feel
## Reference Points (similar songs/artists for inspiration)
"""
    return f"""\
あなたはハル、{genre}専門の作曲家です。
CEOからの指示:
{directive}

楽曲のメロディとハーモニーの基盤を構築してください。

出力形式:
## メロディコンセプト
## コード進行
## 主要モチーフ・フック
## リズム感・グルーヴ
## 参考楽曲（インスピレーション源）
"""


def _arrangement_system(genre: str, language: str, directive: str) -> str:
    if language == "en":
        return f"""\
You are ソラ (Sora), an arranger specializing in {genre}.
The CEO has given you the following directive:
{directive}

Design the full arrangement and instrumentation for the piece.

Output format:
## Instrumentation
## Song Structure (intro, verse, chorus, bridge, outro)
## Texture & Layers
## Dynamic Map
## Sound Design Notes
"""
    return f"""\
あなたはソラ、{genre}専門のアレンジャーです。
CEOからの指示:
{directive}

楽曲の完全な編曲と楽器構成を設計してください。

出力形式:
## 楽器構成
## 楽曲構成（イントロ・Aメロ・Bメロ・サビ・アウトロ）
## テクスチャーとレイヤー
## ダイナミクスマップ
## サウンドデザインメモ
"""


def _production_system(genre: str, language: str, directive: str) -> str:
    if language == "en":
        return f"""\
You are カイ (Kai), a veteran music producer with deep expertise in {genre}.
The CEO has given you the following directive:
{directive}

Review the composition and arrangement, then produce final production notes.

Output:
## Overall Vision Assessment
## Strengths & Refinements
## Mixing & Mastering Direction
## Final Production Blueprint
"""
    return f"""\
あなたはカイ、{genre}に精通したベテランの音楽プロデューサーです。
CEOからの指示:
{directive}

作曲とアレンジをレビューし、最終的な制作指針をまとめてください。

出力:
## 総合ビジョン評価
## 強みと改善点
## ミックス・マスタリング方針
## 最終制作ブループリント
"""


def save_music_file(result: dict, request: str) -> str:
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _safe_filename(request)
    filename = OUTPUT_DIR / f"{timestamp}_{slug}.md"
    lang_label = SUPPORTED_LANGUAGES.get(result.get("language", "ja"), "日本語")
    content = f"""\
# 音楽制作資料 / Music Production Document

**リクエスト:** {request}
**ジャンル:** {result.get('genre', '')}
**言語:** {lang_label}
**作成日時:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## CEO 制作指令書

{result['mandate']}

---

## 作曲資料（ハル）

{result['composition']}

---

## アレンジ資料（ソラ）

{result['arrangement']}

---

## 制作指針（カイ）

{result['production']}
"""
    filename.write_text(content, encoding="utf-8")
    return str(filename)


def create_music(user_request: str) -> dict:
    """CEO→チームエージェントの音楽制作パイプライン。"""
    print("\n👔 [yasu / CEO] 制作指令書を作成中...\n")
    mandate = _stream_response(
        CEO_MUSIC_SYSTEM,
        [{"role": "user", "content": user_request}],
    )

    genre    = _parse_field(mandate, "Genre",         default="J-POP")
    language = _parse_field(mandate, "Language",      default="ja").strip().lower()
    title    = _parse_field(mandate, "Title Concept", default="Untitled")
    if language not in SUPPORTED_LANGUAGES:
        language = "ja"

    lang_label = SUPPORTED_LANGUAGES[language]
    print(f"\n   📋 Genre: {genre} | Language: {lang_label} | Title: {title}")

    comp_dir  = _extract_directive(mandate, "Composition Agent")
    arr_dir   = _extract_directive(mandate, "Arrangement Agent")
    prod_dir  = _extract_directive(mandate, "Production Agent")

    prompt_ja = f"ジャンル: {genre}\nリクエスト: {user_request}"
    prompt_en = f"Genre: {genre}\nRequest: {user_request}"
    base_prompt = prompt_en if language == "en" else prompt_ja

    print(f"\n🎼 [ハル / 作曲エージェント] メロディとコードを構築中...\n")
    composition = _stream_response(
        _composition_system(genre, language, comp_dir),
        [{"role": "user", "content": base_prompt}],
    )

    print(f"\n🎹 [ソラ / アレンジエージェント] 編曲を設計中...\n")
    arrangement_prompt = (
        base_prompt + f"\n\nComposition Brief:\n{composition}"
        if language == "en"
        else base_prompt + f"\n\n作曲資料:\n{composition}"
    )
    arrangement = _stream_response(
        _arrangement_system(genre, language, arr_dir),
        [{"role": "user", "content": arrangement_prompt}],
    )

    print(f"\n🎚️  [カイ / プロデュースエージェント] 制作指針をまとめ中...\n")
    production_prompt = (
        f"Genre: {genre}\nRequest: {user_request}\n\n"
        f"Composition:\n{composition}\n\nArrangement:\n{arrangement}"
        if language == "en"
        else
        f"ジャンル: {genre}\nリクエスト: {user_request}\n\n"
        f"作曲資料:\n{composition}\n\nアレンジ資料:\n{arrangement}"
    )
    production = _stream_response(
        _production_system(genre, language, prod_dir),
        [{"role": "user", "content": production_prompt}],
    )

    result = {
        "mandate":     mandate,
        "genre":       genre,
        "language":    language,
        "title":       title,
        "composition": composition,
        "arrangement": arrangement,
        "production":  production,
    }
    saved = save_music_file(result, user_request)
    result["saved_to"] = saved

    print(f"\n✅ 完成！  Saved → {saved}")
    return result


# ---------------------------------------------------------------------------
# チャット REPL
# ---------------------------------------------------------------------------

def run_chat() -> None:
    print("=" * 60)
    print("🎼 rin.music — 音楽制作チーム")
    print(f'   「{SAVE_KEYWORD}」で保存  「{LOAD_KEYWORD}」で続きから')
    print("=" * 60)

    messages: list = []

    print("\n👔 yasu: ", end="", flush=True)
    greeting = _chat_turn([{"role": "user", "content": "セッション開始。音楽制作チームのCEOとして軽く挨拶して。"}])
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

        if user_input == SAVE_KEYWORD:
            path = save_session(messages)
            print(f"\n💾 会話を保存しました → {path}")
            print("👔 yasu: またね！お疲れ～")
            break

        if user_input == LOAD_KEYWORD:
            loaded = load_session()
            if not loaded:
                print("📂 保存された会話が見つかりませんでした。")
                continue
            messages = loaded
            turns = sum(1 for m in messages if m["role"] == "user")
            print(f"\n📂 前回の会話を読み込みました（{turns}ターン）\n")
            print("👔 yasu: ", end="", flush=True)
            resume_resp = _chat_turn(
                messages + [{"role": "user", "content":
                    "前回の会話の続きです。前回どんな話をしていたか簡単に振り返ってから続けて。"}]
            )
            messages.append({"role": "user",      "content": LOAD_KEYWORD})
            messages.append({"role": "assistant", "content": resume_resp})
            continue

        messages.append({"role": "user", "content": user_input})
        print("\n👔 yasu: ", end="", flush=True)
        response = _chat_turn(messages)

        action_request = _parse_action(response)
        visible = _visible_response(response)

        messages.append({"role": "assistant", "content": visible})

        if action_request:
            print("\n" + "─" * 60)
            result = create_music(action_request)
            print("─" * 60)
            summary = (
                f"[制作完了] タイトル: {result['title']} / "
                f"ジャンル: {result['genre']} / 保存先: {result['saved_to']}"
            )
            messages.append({"role": "user",      "content": "[制作完了の報告]"})
            messages.append({"role": "assistant", "content": summary})


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_chat()
