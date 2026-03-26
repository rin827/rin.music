"""
rin.music — 作詞チームエージェント

4つのエージェントが協力して歌詞を作成します:
  0. CEOエージェント      : リクエストを戦略分析し、制作指令書を作成・チームを指揮
  1. テーマエージェント   : テーマ・世界観・感情を分析・展開
  2. 作詞エージェント    : 歌詞を執筆
  3. レビューエージェント : 歌詞をレビュー・改善提案

機能:
  - 日本語 / English 歌詞対応 (CEOが自動判断)
  - ジャンルをCEOが自動決定 (手動オーバーライド可)
  - 歌詞をファイルに保存
"""

import os
import re
import datetime
import anthropic

MODEL = "claude-opus-4-6"
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SUPPORTED_LANGUAGES = {"ja": "日本語", "en": "English"}

# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _stream_response(system: str, user: str) -> str:
    """ストリーミングでAPIを呼び出し、完全なテキストを返す。"""
    with client.messages.stream(
        model=MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user}],
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
    """テキストからファイル名に使える安全な文字列を生成する。"""
    cleaned = re.sub(r'[^\w\s\-]', '', text, flags=re.UNICODE)
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    return cleaned[:max_len] if cleaned else "lyrics"


def _parse_field(text: str, key: str, default: str = "") -> str:
    """制作指令書から特定フィールドの値を抽出する。"""
    pattern = rf'\*\*{re.escape(key)}\*\*\s*[:\uff1a]\s*(.+)'
    m = re.search(pattern, text)
    return m.group(1).strip() if m else default


def save_lyrics(result: dict, request: str) -> str:
    """
    歌詞と制作指令書をMarkdownファイルに保存する。

    Returns:
        保存したファイルのパス
    """
    output_dir = "lyrics"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _safe_filename(request)
    filename = f"{output_dir}/{timestamp}_{slug}.md"

    mandate = result.get("mandate", "")
    genre = result.get("genre", "")
    language = result.get("language", "ja")
    lang_label = SUPPORTED_LANGUAGES.get(language, language)

    content = f"""\
# 歌詞 / Lyrics

**リクエスト / Request:** {request}
**ジャンル / Genre:** {genre}
**言語 / Language:** {lang_label}
**作成日時:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## CEO 制作指令書 / Production Mandate

{mandate}

---

## テーマ資料 / Theme Brief

{result['theme']}

---

## 初稿 / Draft

{result['draft']}

---

## レビュー & 最終版 / Review & Final

{result['final']}
"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    return filename


# ---------------------------------------------------------------------------
# システムプロンプト
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
You are a music producer and creative director specializing in {genre}.
The CEO has given you the following directive:
{directive}

Analyze the request and develop the theme, world, emotions, and structure for the lyrics.

Output format:
## Theme
(Core theme in 1-2 sentences)

## World & Imagery
(Scene and imagery to portray)

## Emotional Arc
(How emotions evolve throughout the song)

## Keywords & Phrases
(Candidate images and phrases for the lyrics)

## Structure Suggestion
(Role of Verse / Pre-Chorus / Chorus / Bridge etc.)
"""
    else:
        return f"""\
あなたは{genre}専門の音楽プロデューサー兼クリエイティブディレクターです。
CEOからの指示:
{directive}

リクエストをもとに、歌詞のテーマ・世界観・感情・構成を深く掘り下げてください。

出力形式:
## テーマ
(コアとなるテーマを1〜2文で)

## 世界観・情景
(どんな場面・情景を描くか)

## 感情の流れ
(曲を通じて感情がどう変化するか)

## キーワード
(歌詞に使いたいイメージやフレーズの候補)

## 構成提案
(Aメロ・Bメロ・サビ・Cメロなどの役割)
"""


def _writer_system(genre: str, language: str, directive: str) -> str:
    if language == "en":
        return f"""\
You are a professional lyricist specializing in {genre}.
The CEO has given you the following directive:
{directive}

Write complete, polished English lyrics based on the theme brief provided.

Guidelines:
- Choose words that convey emotion naturally
- Craft lines that feel good to sing (rhythm, syllable flow)
- Use vivid metaphors and imagery
- Label each section clearly (Verse 1, Pre-Chorus, Chorus, Verse 2, Bridge, Outro, etc.)
- Ensure hooks are memorable and genre-appropriate for {genre}
"""
    else:
        return f"""\
あなたは{genre}専門のプロの作詞家です。
CEOからの指示:
{directive}

テーマエージェントが作成したテーマ資料をもとに、完成度の高い日本語の歌詞を執筆してください。

ガイドライン:
- 感情が自然に伝わる言葉を選ぶ
- 歌として口ずさんだとき気持ちよいリズム感を意識する
- 比喩・情景描写を効果的に使う
- Aメロ・Bメロ・サビ・(Cメロ) などの構成を明示する
- {genre}らしいフレーズ・言葉選びを意識する
"""


def _review_system(genre: str, language: str, directive: str) -> str:
    if language == "en":
        return f"""\
You are a veteran music director with deep expertise in {genre}.
The CEO has given you the following directive:
{directive}

Review and improve the draft lyrics provided.

Evaluation criteria:
1. Consistency with the theme and CEO mandate
2. Lyrical rhythm and flow
3. Emotional intensity and authenticity
4. Memorable hooks and phrasing
5. Genre fit for {genre}

Output:
## Overall Assessment
(3-4 sentences)

## Points to Improve
(bullet points)

## Revised Lyrics
(complete revised lyrics)
"""
    else:
        return f"""\
あなたは{genre}に精通したベテランの音楽ディレクターです。
CEOからの指示:
{directive}

作詞エージェントが書いた歌詞を批評・改善してください。

評価観点:
1. テーマ・CEO指令との一貫性
2. 言葉のリズム・流れ
3. 感情の強度と自然さ
4. 印象的なフレーズ・フック
5. {genre}としての雰囲気・適合度

出力:
## 総評
(3〜4文)

## 改善ポイント
(箇条書き)

## 改善版歌詞
(修正済みの完全な歌詞)
"""


# ---------------------------------------------------------------------------
# エージェント
# ---------------------------------------------------------------------------

def _extract_directive(mandate: str, agent_name: str) -> str:
    """制作指令書から特定エージェントへの指示を抽出する。"""
    pattern = rf'### Directive[^#]*?{re.escape(agent_name)}.*?\n(.*?)(?=###|\Z)'
    m = re.search(pattern, mandate, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def ceo_agent(user_request: str) -> dict:
    """
    CEOエージェント: 戦略分析と制作指令書の作成。

    Returns:
        {
            "mandate":  str,  # 制作指令書のフルテキスト
            "genre":    str,  # 決定したジャンル
            "language": str,  # 決定した言語 ("ja" or "en")
            "title":    str,  # タイトル案
        }
    """
    print("\n👔 [yasu / CEO] 戦略を策定中 / Analyzing request...\n")
    mandate = _stream_response(CEO_SYSTEM, f"User request: {user_request}")

    genre    = _parse_field(mandate, "Genre",         default="J-POP Ballad")
    language = _parse_field(mandate, "Language",      default="ja").strip().lower()
    title    = _parse_field(mandate, "Title Concept", default="Untitled")

    if language not in SUPPORTED_LANGUAGES:
        language = "ja"

    return {"mandate": mandate, "genre": genre, "language": language, "title": title}


def theme_agent(user_request: str, genre: str, language: str, directive: str) -> str:
    label = "🎵 [Theme Agent]" if language == "en" else "🎵 [テーマエージェント]"
    msg = "Building world and theme..." if language == "en" else "世界観を構築中..."
    print(f"\n{label} {msg}\n")
    prompt = (
        f"Genre: {genre}\n\nRequest: {user_request}"
        if language == "en"
        else f"ジャンル: {genre}\n\nリクエスト: {user_request}"
    )
    return _stream_response(_theme_system(genre, language, directive), prompt)


def lyric_writer_agent(
    user_request: str, theme_brief: str, genre: str, language: str, directive: str
) -> str:
    label = "✍️  [Lyric Writer]" if language == "en" else "✍️  [作詞エージェント]"
    msg = "Writing lyrics..." if language == "en" else "歌詞を書いています..."
    print(f"\n{label} {msg}\n")
    prompt = (
        f"Genre: {genre}\nRequest: {user_request}\n\nTheme Brief:\n{theme_brief}\n\n"
        "Write the full lyrics based on the above."
        if language == "en"
        else (
            f"ジャンル: {genre}\nリクエスト: {user_request}\n\n"
            f"テーマ資料:\n{theme_brief}\n\n上記をもとに歌詞を書いてください。"
        )
    )
    return _stream_response(_writer_system(genre, language, directive), prompt)


def review_agent(
    user_request: str, theme_brief: str, draft_lyrics: str,
    genre: str, language: str, directive: str
) -> str:
    label = "🔍 [Review Agent]" if language == "en" else "🔍 [レビューエージェント]"
    msg = "Reviewing lyrics..." if language == "en" else "歌詞をレビュー中..."
    print(f"\n{label} {msg}\n")
    prompt = (
        f"Genre: {genre}\nOriginal Request: {user_request}\n\n"
        f"Theme Brief:\n{theme_brief}\n\nDraft Lyrics:\n{draft_lyrics}\n\n"
        "Critique and provide an improved version."
        if language == "en"
        else (
            f"ジャンル: {genre}\n元リクエスト: {user_request}\n\n"
            f"テーマ資料:\n{theme_brief}\n\n歌詞草稿:\n{draft_lyrics}\n\n"
            "この歌詞を批評し、改善版を提供してください。"
        )
    )
    return _stream_response(_review_system(genre, language, directive), prompt)


# ---------------------------------------------------------------------------
# オーケストレーター
# ---------------------------------------------------------------------------

def create_lyrics(user_request: str) -> dict:
    """
    CEOが戦略を立て、3エージェントを指揮して歌詞を作成・保存する。

    Args:
        user_request: 曲のイメージ・リクエスト (ジャンル・言語はCEOが自動決定)

    Returns:
        {"mandate": str, "genre": str, "language": str, "title": str,
         "theme": str, "draft": str, "final": str, "saved_to": str}
    """
    print("=" * 60)
    print("🎼 rin.music — Lyrics Team Agent")
    print("=" * 60)

    # Step 0: CEO が戦略決定
    ceo = ceo_agent(user_request)
    genre    = ceo["genre"]
    language = ceo["language"]
    mandate  = ceo["mandate"]

    lang_label = SUPPORTED_LANGUAGES.get(language, language)
    print(f"\n   📋 Genre: {genre} | Language: {lang_label} | Title: {ceo['title']}")

    # Step 1–3: チームエージェントが制作
    theme_dir  = _extract_directive(mandate, "Theme Agent")
    writer_dir = _extract_directive(mandate, "Lyric Writer")
    review_dir = _extract_directive(mandate, "Review Agent")

    theme_brief  = theme_agent(user_request, genre, language, theme_dir)
    draft_lyrics = lyric_writer_agent(user_request, theme_brief, genre, language, writer_dir)
    final_review = review_agent(user_request, theme_brief, draft_lyrics, genre, language, review_dir)

    result = {**ceo, "theme": theme_brief, "draft": draft_lyrics, "final": final_review}

    saved_path = save_lyrics(result, user_request)
    result["saved_to"] = saved_path

    print("\n" + "=" * 60)
    print(f"✅ 完成！  Saved → {saved_path}")
    print("=" * 60)

    return result


# ---------------------------------------------------------------------------
# CLI エントリーポイント
# ---------------------------------------------------------------------------

def _ask(prompt: str, default: str = "") -> str:
    value = input(prompt).strip()
    return value if value else default


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("🎼 rin.music — Lyrics Team Agent")
    print("   (ジャンル・言語はCEOが自動決定します)")
    print("=" * 60)

    if len(sys.argv) > 1:
        request = " ".join(sys.argv[1:])
    else:
        request = _ask(
            "\n曲のイメージを自由に入力してください\n"
            "  例: 夏の終わりの切ない恋愛\n"
            "  例: An upbeat song about chasing dreams in a big city\n"
            "> ",
            "夏の終わり、遠距離恋愛、会いたい気持ち",
        )

    create_lyrics(request)
