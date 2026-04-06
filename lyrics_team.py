"""
rin.music — 作詞チームエージェント

メンバー:
  👔 yasu         : CEO — 戦略・指揮
  🎵 龍姫（たつき）: テーマエージェント
  ✍️  レイ          : 作詞エージェント
  🔍 ルキ          : レビューエージェント

スタジオチーム:
  🎸 ケンジ        : アレンジャー — コード・楽器編成
  🎤 ミオ          : ボイスディレクター — ボーカル指導
  🎚️  ダイ          : エンジニア — 録音・ミックス設定

使い方:
  python lyrics_team.py       チャットモードで起動
  「おつ～」                   会話を保存して終了
  「よろ～」                   前回の会話を読み込んで続きから
"""

import os
import re
import json
import datetime
import anthropic
from pathlib import Path

MODEL   = "claude-opus-4-6"
client  = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SUPPORTED_LANGUAGES = {"ja": "日本語", "en": "English"}

SAVE_KEYWORD = "おつ～"
LOAD_KEYWORD = "よろ～"
SESSION_DIR  = Path("sessions")
SESSION_FILE = SESSION_DIR / "latest.json"

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
    return cleaned[:max_len] if cleaned else "lyrics"


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

YASU_CHAT_SYSTEM = """\
You are yasu, the CEO of rin.music, a creative music production company.
You're having a natural, casual conversation with your client about their music.

Your lyrics team:
- 龍姫（たつき）: Theme Agent — develops themes, worldview, emotional arcs
- レイ: Lyric Writer — writes the actual lyrics
- ルキ: Review Agent — critiques and refines lyrics

Your studio team:
- ケンジ: Arranger — chord progressions, instrumentation, BPM, key
- ミオ: Voice Director — vocal delivery, emotion, phrasing guidance
- ダイ: Sound Engineer — recording setup, mic, mixing notes

Personality: creative, enthusiastic, concise. You speak casually but with professional instincts.
Always reply in the same language the user uses (Japanese or English).

When the user wants lyrics ACTUALLY CREATED (not just discussed), \
output this line at the very end of your response — nothing after it:
ACTION: CREATE_LYRICS | <one-line summary of what to create>

When the user wants to START A RECORDING SESSION or enter the studio \
(e.g. they have lyrics ready and want to record, or say スタジオイン / studio in / let's record), \
output this line at the very end of your response — nothing after it:
ACTION: STUDIO_IN | <one-line summary of the song and session goal>

Only add ONE ACTION line when you are ready. \
Do NOT add it for conceptual discussions or questions.
"""


def _chat_turn(messages: list) -> str:
    """yasuとのチャットターン（thinking なし・短めのトークン）。"""
    with client.messages.stream(
        model=MODEL,
        max_tokens=1024,
        system=YASU_CHAT_SYSTEM,
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
    """ACTION: CREATE_LYRICS | <request> を抽出する。なければ None。"""
    m = re.search(r'ACTION:\s*CREATE_LYRICS\s*\|\s*(.+)', response)
    return m.group(1).strip() if m else None


def _parse_studio_action(response: str) -> str | None:
    """ACTION: STUDIO_IN | <request> を抽出する。なければ None。"""
    m = re.search(r'ACTION:\s*STUDIO_IN\s*\|\s*(.+)', response)
    return m.group(1).strip() if m else None


def _visible_response(response: str) -> str:
    """ACTION行をユーザーに見せないよう除去したテキスト。"""
    cleaned = re.sub(r'\nACTION:\s*CREATE_LYRICS\s*\|.+', '', response)
    cleaned = re.sub(r'\nACTION:\s*STUDIO_IN\s*\|.+', '', cleaned)
    return cleaned.rstrip()


# ---------------------------------------------------------------------------
# 作詞パイプライン（CEO→チーム）
# ---------------------------------------------------------------------------

CEO_SYSTEM = """\
You are yasu, the CEO of rin.music, a creative music production company.
Your job is to receive a user's song request and produce a strategic production mandate \
that directs your team of agents (Theme Agent, Lyric Writer, Review Agent).

Analyze the request and decide:
- Genre (be specific, e.g. "J-POP Ballad", "City Pop", "Indie Folk", "R&B", "Rock")
- Language ("ja" for Japanese lyrics, "en" for English lyrics) — \
  infer from the request language and content; default to "ja" if ambiguous
- Title concept (a working title for the song)
- Target audience
- Emotional goal
- Tone and style
- Specific directives for each team member

Output your mandate in the following exact format (fill in every field):

**Genre:** <genre>
**Language:** <ja or en>
**Title Concept:** <working title>
**Target Audience:** <audience>
**Emotional Goal:** <goal>
**Tone:** <tone and style>

### Directive — Theme Agent
<instructions for the theme agent>

### Directive — Lyric Writer
<instructions for the lyric writer>

### Directive — Review Agent
<instructions for the review agent>

### CEO Note
<any additional strategic notes or creative vision>
"""


def _theme_system(genre: str, language: str, directive: str) -> str:
    if language == "en":
        return f"""\
You are 龍姫 (Tatsuki), a music producer and creative director specializing in {genre}.
The CEO has given you the following directive:
{directive}

Develop the theme, world, emotions, and structure for the lyrics.

Output format:
## Theme
## World & Imagery
## Emotional Arc
## Keywords & Phrases
## Structure Suggestion
"""
    return f"""\
あなたは龍姫（たつき）、{genre}専門の音楽プロデューサー兼クリエイティブディレクターです。
CEOからの指示:
{directive}

出力形式:
## テーマ
## 世界観・情景
## 感情の流れ
## キーワード
## 構成提案
"""


def _writer_system(genre: str, language: str, directive: str) -> str:
    if language == "en":
        return f"""\
You are レイ, a professional lyricist specializing in {genre}.
The CEO has given you the following directive:
{directive}

Write complete, polished English lyrics. Label each section clearly.
"""
    return f"""\
あなたはレイ、{genre}専門のプロの作詞家です。
CEOからの指示:
{directive}

完成度の高い日本語の歌詞を執筆してください。Aメロ・Bメロ・サビなど構成を明示すること。
"""


def _review_system(genre: str, language: str, directive: str) -> str:
    if language == "en":
        return f"""\
You are ルキ, a veteran music director with deep expertise in {genre}.
The CEO has given you the following directive:
{directive}

Output:
## Overall Assessment
## Points to Improve
## Revised Lyrics
"""
    return f"""\
あなたはルキ、{genre}に精通したベテランの音楽ディレクターです。
CEOからの指示:
{directive}

出力:
## 総評
## 改善ポイント
## 改善版歌詞
"""


def save_lyrics_file(result: dict, request: str) -> str:
    output_dir = Path("lyrics")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _safe_filename(request)
    filename = output_dir / f"{timestamp}_{slug}.md"
    lang_label = SUPPORTED_LANGUAGES.get(result.get("language", "ja"), "日本語")
    content = f"""\
# 歌詞 / Lyrics

**リクエスト:** {request}
**ジャンル:** {result.get('genre', '')}
**言語:** {lang_label}
**作成日時:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## CEO 制作指令書

{result['mandate']}

---

## テーマ資料

{result['theme']}

---

## 初稿

{result['draft']}

---

## レビュー & 最終版

{result['final']}
"""
    filename.write_text(content, encoding="utf-8")
    return str(filename)


def create_lyrics(user_request: str) -> dict:
    """CEO→チームエージェントの作詞パイプライン。"""
    print("\n👔 [yasu / CEO] 制作指令書を作成中...\n")
    mandate = _stream_response(CEO_SYSTEM, [{"role": "user", "content": user_request}])

    genre    = _parse_field(mandate, "Genre",         default="J-POP Ballad")
    language = _parse_field(mandate, "Language",      default="ja").strip().lower()
    title    = _parse_field(mandate, "Title Concept", default="Untitled")
    if language not in SUPPORTED_LANGUAGES:
        language = "ja"

    lang_label = SUPPORTED_LANGUAGES[language]
    print(f"\n   📋 Genre: {genre} | Language: {lang_label} | Title: {title}")

    theme_dir  = _extract_directive(mandate, "Theme Agent")
    writer_dir = _extract_directive(mandate, "Lyric Writer")
    review_dir = _extract_directive(mandate, "Review Agent")

    prompt_ja = f"ジャンル: {genre}\nリクエスト: {user_request}"
    prompt_en = f"Genre: {genre}\nRequest: {user_request}"
    base_prompt = prompt_en if language == "en" else prompt_ja

    print(f"\n🎵 [龍姫 / テーマエージェント] 世界観を構築中...\n")
    theme_brief = _stream_response(
        _theme_system(genre, language, theme_dir),
        [{"role": "user", "content": base_prompt}],
    )

    print(f"\n✍️  [レイ / 作詞エージェント] 歌詞を書いています...\n")
    writer_prompt = base_prompt + f"\n\nTheme Brief:\n{theme_brief}" \
        if language == "en" else base_prompt + f"\n\nテーマ資料:\n{theme_brief}"
    draft_lyrics = _stream_response(
        _writer_system(genre, language, writer_dir),
        [{"role": "user", "content": writer_prompt}],
    )

    print(f"\n🔍 [ルキ / レビューエージェント] 歌詞をレビュー中...\n")
    review_prompt = (
        f"Genre: {genre}\nRequest: {user_request}\n\n"
        f"Theme:\n{theme_brief}\n\nDraft:\n{draft_lyrics}"
        if language == "en" else
        f"ジャンル: {genre}\nリクエスト: {user_request}\n\n"
        f"テーマ資料:\n{theme_brief}\n\n歌詞草稿:\n{draft_lyrics}"
    )
    final_review = _stream_response(
        _review_system(genre, language, review_dir),
        [{"role": "user", "content": review_prompt}],
    )

    result = {
        "mandate": mandate, "genre": genre, "language": language, "title": title,
        "theme": theme_brief, "draft": draft_lyrics, "final": final_review,
    }
    saved = save_lyrics_file(result, user_request)
    result["saved_to"] = saved

    print(f"\n✅ 完成！  Saved → {saved}")
    return result


# ---------------------------------------------------------------------------
# スタジオインパイプライン（アレンジャー→ボイスディレクター→エンジニア）
# ---------------------------------------------------------------------------

STUDIO_CEO_SYSTEM = """\
You are yasu, the CEO of rin.music.
Your job is to receive a recording session request and produce a Studio Production Brief \
that directs your studio team (Arranger, Voice Director, Sound Engineer).

Analyze the request and decide:
- Genre (be specific)
- Language ("ja" or "en")
- Song Title (working title or given title)
- BPM Range (suggested tempo)
- Key (suggested musical key)
- Session Goal (what this session aims to achieve)
- Specific directives for each studio team member

Output your brief in the following exact format (fill in every field):

**Genre:** <genre>
**Language:** <ja or en>
**Song Title:** <title>
**BPM Range:** <e.g. 80-90>
**Key:** <e.g. A minor>
**Session Goal:** <goal>

### Directive — Arranger
<instructions for the arranger>

### Directive — Voice Director
<instructions for the voice director>

### Directive — Sound Engineer
<instructions for the sound engineer>

### CEO Studio Note
<any additional creative or strategic notes>
"""


def _arranger_system(genre: str, language: str, directive: str) -> str:
    if language == "en":
        return f"""\
You are ケンジ (Kenji), a professional music arranger specializing in {genre}.
The CEO has given you the following directive:
{directive}

Create a detailed arrangement sheet for this recording session.

Output format:
## Chord Progression
## Instrumentation
## Structure & Sections
## BPM & Groove
## Reference Tracks
"""
    return f"""\
あなたはケンジ、{genre}専門のプロのアレンジャーです。
CEOからの指示:
{directive}

このレコーディングセッション用の詳細なアレンジシートを作成してください。

出力形式:
## コード進行
## 楽器編成
## 構成・セクション
## BPMとグルーヴ
## 参考楽曲
"""


def _voice_director_system(genre: str, language: str, directive: str) -> str:
    if language == "en":
        return f"""\
You are ミオ (Mio), a vocal coach and voice director specializing in {genre}.
The CEO has given you the following directive:
{directive}

Create a detailed vocal direction sheet for this recording session.

Output format:
## Overall Vocal Tone
## Section-by-Section Delivery Guide
## Breath & Phrasing Notes
## Emotion & Expression Cues
## Warm-up Suggestions
"""
    return f"""\
あなたはミオ、{genre}専門のボイスコーチ兼ボイスディレクターです。
CEOからの指示:
{directive}

このレコーディングセッション用の詳細なボーカルディレクションシートを作成してください。

出力形式:
## 全体のボーカルトーン
## セクション別歌い方ガイド
## ブレスとフレージング
## 感情・表現のキュー
## ウォームアップ提案
"""


def _engineer_system(genre: str, language: str, directive: str) -> str:
    if language == "en":
        return f"""\
You are ダイ (Dai), a recording and mixing engineer specializing in {genre}.
The CEO has given you the following directive:
{directive}

Create a technical setup sheet for this recording session.

Output format:
## Microphone Setup
## Signal Chain & Preamp Notes
## Room & Acoustics
## Monitoring Setup
## Mix Direction & Key Processing
"""
    return f"""\
あなたはダイ、{genre}専門のレコーディング＆ミックスエンジニアです。
CEOからの指示:
{directive}

このレコーディングセッション用のテクニカルセットアップシートを作成してください。

出力形式:
## マイクセットアップ
## シグナルチェーン・プリアンプ
## 部屋・アコースティクス
## モニタリング設定
## ミックス方針・主要処理
"""


def save_studio_file(result: dict, request: str) -> str:
    output_dir = Path("studio")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _safe_filename(request)
    filename = output_dir / f"{timestamp}_{slug}.md"
    lang_label = SUPPORTED_LANGUAGES.get(result.get("language", "ja"), "日本語")
    content = f"""\
# スタジオシート / Studio Sheet

**セッション:** {request}
**ジャンル:** {result.get('genre', '')}
**言語:** {lang_label}
**タイトル:** {result.get('title', '')}
**BPM範囲:** {result.get('bpm', '')}
**キー:** {result.get('key', '')}
**作成日時:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## CEO スタジオ制作指令書

{result['brief']}

---

## アレンジシート（ケンジ）

{result['arrangement']}

---

## ボーカルディレクション（ミオ）

{result['vocal_direction']}

---

## テクニカルセットアップ（ダイ）

{result['engineering']}
"""
    filename.write_text(content, encoding="utf-8")
    return str(filename)


def create_studio_session(user_request: str) -> dict:
    """スタジオインパイプライン: CEO→アレンジャー→ボイスディレクター→エンジニア。"""
    print("\n👔 [yasu / CEO] スタジオ制作指令書を作成中...\n")
    brief = _stream_response(
        STUDIO_CEO_SYSTEM,
        [{"role": "user", "content": user_request}],
    )

    genre    = _parse_field(brief, "Genre",        default="J-POP")
    language = _parse_field(brief, "Language",     default="ja").strip().lower()
    title    = _parse_field(brief, "Song Title",   default="Untitled")
    bpm      = _parse_field(brief, "BPM Range",    default="")
    key      = _parse_field(brief, "Key",          default="")
    if language not in SUPPORTED_LANGUAGES:
        language = "ja"

    lang_label = SUPPORTED_LANGUAGES[language]
    print(f"\n   📋 Genre: {genre} | Language: {lang_label} | Title: {title} | BPM: {bpm} | Key: {key}")

    arr_dir   = _extract_directive(brief, "Arranger")
    voice_dir = _extract_directive(brief, "Voice Director")
    eng_dir   = _extract_directive(brief, "Sound Engineer")

    base_prompt = (
        f"Genre: {genre}\nSession request: {user_request}"
        if language == "en" else
        f"ジャンル: {genre}\nセッションリクエスト: {user_request}"
    )

    print(f"\n🎸 [ケンジ / アレンジャー] アレンジシートを作成中...\n")
    arrangement = _stream_response(
        _arranger_system(genre, language, arr_dir),
        [{"role": "user", "content": base_prompt}],
    )

    print(f"\n🎤 [ミオ / ボイスディレクター] ボーカルディレクションを作成中...\n")
    voice_prompt = (
        base_prompt + f"\n\nArrangement:\n{arrangement}"
        if language == "en" else
        base_prompt + f"\n\nアレンジ:\n{arrangement}"
    )
    vocal_direction = _stream_response(
        _voice_director_system(genre, language, voice_dir),
        [{"role": "user", "content": voice_prompt}],
    )

    print(f"\n🎚️  [ダイ / エンジニア] テクニカルセットアップを作成中...\n")
    eng_prompt = (
        base_prompt + f"\n\nArrangement:\n{arrangement}"
        if language == "en" else
        base_prompt + f"\n\nアレンジ:\n{arrangement}"
    )
    engineering = _stream_response(
        _engineer_system(genre, language, eng_dir),
        [{"role": "user", "content": eng_prompt}],
    )

    result = {
        "brief": brief, "genre": genre, "language": language, "title": title,
        "bpm": bpm, "key": key,
        "arrangement": arrangement,
        "vocal_direction": vocal_direction,
        "engineering": engineering,
    }
    saved = save_studio_file(result, user_request)
    result["saved_to"] = saved

    print(f"\n✅ スタジオシート完成！  Saved → {saved}")
    return result


# ---------------------------------------------------------------------------
# チャット REPL
# ---------------------------------------------------------------------------

def run_chat() -> None:
    print("=" * 60)
    print("🎼 rin.music")
    print(f'   「{SAVE_KEYWORD}」で保存  「{LOAD_KEYWORD}」で続きから')
    print("=" * 60)

    messages: list = []

    # 開幕挨拶
    print("\n👔 yasu: ", end="", flush=True)
    greeting = _chat_turn([{"role": "user", "content": "セッション開始。軽く挨拶して。"}])
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
            path = save_session(messages)
            print(f"\n💾 会話を保存しました → {path}")
            print("👔 yasu: またね！お疲れ～")
            break

        # ── 読み込み ──
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

        # ── 通常会話 ──
        messages.append({"role": "user", "content": user_input})
        print("\n👔 yasu: ", end="", flush=True)
        response = _chat_turn(messages)

        action_request  = _parse_action(response)
        studio_request  = _parse_studio_action(response)
        visible = _visible_response(response)

        messages.append({"role": "assistant", "content": visible})

        # ── 作詞アクション ──
        if action_request:
            print("\n" + "─" * 60)
            result = create_lyrics(action_request)
            print("─" * 60)
            summary = (
                f"[作詞完了] タイトル: {result['title']} / "
                f"ジャンル: {result['genre']} / 保存先: {result['saved_to']}"
            )
            messages.append({"role": "user",      "content": "[作詞完了の報告]"})
            messages.append({"role": "assistant", "content": summary})

        # ── スタジオインアクション ──
        elif studio_request:
            print("\n" + "─" * 60)
            result = create_studio_session(studio_request)
            print("─" * 60)
            summary = (
                f"[スタジオイン完了] タイトル: {result['title']} / "
                f"ジャンル: {result['genre']} / 保存先: {result['saved_to']}"
            )
            messages.append({"role": "user",      "content": "[スタジオイン完了の報告]"})
            messages.append({"role": "assistant", "content": summary})


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_chat()
