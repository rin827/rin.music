"""
rin.music — 作詞チームエージェント

3つのエージェントが協力して歌詞を作成します:
  1. テーマエージェント   : テーマ・世界観・感情を分析・展開
  2. 作詞エージェント    : 歌詞を執筆
  3. レビューエージェント : 歌詞をレビュー・改善提案

機能:
  - 日本語 / English 歌詞対応
  - ジャンルを都度設定
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


def save_lyrics(result: dict, request: str, genre: str, language: str) -> str:
    """
    歌詞をMarkdownファイルに保存する。

    Returns:
        保存したファイルのパス
    """
    output_dir = "lyrics"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _safe_filename(request)
    filename = f"{output_dir}/{timestamp}_{slug}.md"

    lang_label = SUPPORTED_LANGUAGES.get(language, language)
    content = f"""\
# 歌詞 / Lyrics

**リクエスト / Request:** {request}
**ジャンル / Genre:** {genre}
**言語 / Language:** {lang_label}
**作成日時:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

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
# システムプロンプト（言語・ジャンル対応）
# ---------------------------------------------------------------------------

def _theme_system(genre: str, language: str) -> str:
    if language == "en":
        return f"""\
You are a music producer and creative director specializing in {genre}.
Analyze the user's request and develop the theme, world, emotions, and structure for the lyrics.

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
ユーザーのリクエストをもとに、歌詞のテーマ・世界観・感情・構成を深く掘り下げてください。

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


def _writer_system(genre: str, language: str) -> str:
    if language == "en":
        return f"""\
You are a professional lyricist specializing in {genre}.
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
テーマエージェントが作成したテーマ資料をもとに、完成度の高い日本語の歌詞を執筆してください。

ガイドライン:
- 感情が自然に伝わる言葉を選ぶ
- 歌として口ずさんだとき気持ちよいリズム感を意識する
- 比喩・情景描写を効果的に使う
- Aメロ・Bメロ・サビ・(Cメロ) などの構成を明示する
- {genre}らしいフレーズ・言葉選びを意識する
"""


def _review_system(genre: str, language: str) -> str:
    if language == "en":
        return f"""\
You are a veteran music director with deep expertise in {genre}.
Review and improve the draft lyrics provided.

Evaluation criteria:
1. Consistency with the theme
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
作詞エージェントが書いた歌詞を批評・改善してください。

評価観点:
1. テーマとの一貫性
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

def theme_agent(user_request: str, genre: str, language: str) -> str:
    label = "🎵 [Theme Agent]" if language == "en" else "🎵 [テーマエージェント]"
    msg = "Building world and theme..." if language == "en" else "世界観を構築中..."
    print(f"\n{label} {msg}\n")
    prompt = f"Genre: {genre}\n\nRequest: {user_request}" if language == "en" \
        else f"ジャンル: {genre}\n\nリクエスト: {user_request}"
    return _stream_response(_theme_system(genre, language), prompt)


def lyric_writer_agent(
    user_request: str, theme_brief: str, genre: str, language: str
) -> str:
    label = "✍️  [Lyric Writer]" if language == "en" else "✍️  [作詞エージェント]"
    msg = "Writing lyrics..." if language == "en" else "歌詞を書いています..."
    print(f"\n{label} {msg}\n")
    if language == "en":
        prompt = (
            f"Genre: {genre}\nRequest: {user_request}\n\n"
            f"Theme Brief:\n{theme_brief}\n\n"
            "Write the full lyrics based on the above."
        )
    else:
        prompt = (
            f"ジャンル: {genre}\nリクエスト: {user_request}\n\n"
            f"テーマ資料:\n{theme_brief}\n\n"
            "上記をもとに歌詞を書いてください。"
        )
    return _stream_response(_writer_system(genre, language), prompt)


def review_agent(
    user_request: str, theme_brief: str, draft_lyrics: str, genre: str, language: str
) -> str:
    label = "🔍 [Review Agent]" if language == "en" else "🔍 [レビューエージェント]"
    msg = "Reviewing lyrics..." if language == "en" else "歌詞をレビュー中..."
    print(f"\n{label} {msg}\n")
    if language == "en":
        prompt = (
            f"Genre: {genre}\nOriginal Request: {user_request}\n\n"
            f"Theme Brief:\n{theme_brief}\n\n"
            f"Draft Lyrics:\n{draft_lyrics}\n\n"
            "Critique and provide an improved version."
        )
    else:
        prompt = (
            f"ジャンル: {genre}\n元リクエスト: {user_request}\n\n"
            f"テーマ資料:\n{theme_brief}\n\n"
            f"歌詞草稿:\n{draft_lyrics}\n\n"
            "この歌詞を批評し、改善版を提供してください。"
        )
    return _stream_response(_review_system(genre, language), prompt)


# ---------------------------------------------------------------------------
# オーケストレーター
# ---------------------------------------------------------------------------

def create_lyrics(user_request: str, genre: str, language: str = "ja") -> dict:
    """
    3エージェントを順番に実行し、歌詞を作成してファイルに保存する。

    Args:
        user_request: 曲のイメージ・リクエスト
        genre:        ジャンル (例: "J-POPバラード", "City Pop", "Rock")
        language:     "ja" (日本語) または "en" (English)

    Returns:
        {"theme": str, "draft": str, "final": str, "saved_to": str}
    """
    lang_label = SUPPORTED_LANGUAGES.get(language, language)
    print("=" * 60)
    print(f"🎼 rin.music — Lyrics Team Agent")
    print(f"   Genre: {genre} | Language: {lang_label}")
    print("=" * 60)

    theme_brief  = theme_agent(user_request, genre, language)
    draft_lyrics = lyric_writer_agent(user_request, theme_brief, genre, language)
    final_review = review_agent(user_request, theme_brief, draft_lyrics, genre, language)

    result = {"theme": theme_brief, "draft": draft_lyrics, "final": final_review}

    saved_path = save_lyrics(result, user_request, genre, language)
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


def _choose_language() -> str:
    print("\n言語 / Language:")
    print("  [1] 日本語 (ja)")
    print("  [2] English (en)")
    choice = _ask("選択 / Choice [1]: ", "1")
    return "en" if choice == "2" else "ja"


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("🎼 rin.music — Lyrics Team Agent")
    print("=" * 60)

    # --- 言語選択 ---
    language = _choose_language()
    lang_label = SUPPORTED_LANGUAGES[language]

    # --- ジャンル ---
    genre_prompt = (
        "Genre (e.g. Pop Ballad, R&B, Rock, City Pop): "
        if language == "en"
        else "ジャンル (例: J-POPバラード, シティポップ, ロック, R&B): "
    )
    genre = _ask(f"\n{genre_prompt}", "J-POPバラード" if language == "ja" else "Pop Ballad")

    # --- リクエスト ---
    if len(sys.argv) > 1:
        request = " ".join(sys.argv[1:])
    else:
        req_prompt = (
            "Describe your song (theme, mood, story): "
            if language == "en"
            else "曲のイメージを入力してください (テーマ・気分・ストーリーなど): "
        )
        default_req = (
            "Late summer, long-distance love, longing to meet"
            if language == "en"
            else "夏の終わり、遠距離恋愛、会いたい気持ち"
        )
        request = _ask(f"\n{req_prompt}", default_req)

    create_lyrics(request, genre, language)
