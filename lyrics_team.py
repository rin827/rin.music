"""
rin.music — 作詞チームエージェント

3つのエージェントが協力して歌詞を作成します:
  1. テーマエージェント   : テーマ・世界観・感情を分析・展開
  2. 作詞エージェント    : 歌詞を執筆
  3. レビューエージェント : 歌詞をレビュー・改善提案
"""

import os
import anthropic

MODEL = "claude-opus-4-6"
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


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
        print()  # 改行
        return "".join(chunks)


# ---------------------------------------------------------------------------
# エージェント定義
# ---------------------------------------------------------------------------

THEME_SYSTEM = """\
あなたは音楽プロデューサー兼クリエイティブディレクターです。
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
(Aメロ・Bメロ・サビなどの役割)
"""

WRITER_SYSTEM = """\
あなたはプロの作詞家です。
テーマエージェントが作成したテーマ資料をもとに、完成度の高い日本語の歌詞を執筆してください。

ガイドライン:
- 感情が自然に伝わる言葉を選ぶ
- 歌として口ずさんだとき気持ちよいリズム感を意識する
- 比喩・情景描写を効果的に使う
- Aメロ・Bメロ・サビ・(Cメロ) などの構成を明示する
"""

REVIEW_SYSTEM = """\
あなたはベテランの音楽ディレクターです。
作詞エージェントが書いた歌詞を批評・改善してください。

評価観点:
1. テーマとの一貫性
2. 言葉のリズム・流れ
3. 感情の強度と自然さ
4. 印象的なフレーズ・フック

出力:
## 総評
(3〜4文)

## 改善ポイント
(箇条書き)

## 改善版歌詞
(修正済みの完全な歌詞)
"""


def theme_agent(user_request: str) -> str:
    """テーマ・世界観を展開する。"""
    print("\n🎵 [テーマエージェント] 世界観を構築中...\n")
    return _stream_response(THEME_SYSTEM, user_request)


def lyric_writer_agent(user_request: str, theme_brief: str) -> str:
    """テーマをもとに歌詞を執筆する。"""
    print("\n✍️  [作詞エージェント] 歌詞を書いています...\n")
    prompt = (
        f"ユーザーのリクエスト:\n{user_request}\n\n"
        f"テーマ資料:\n{theme_brief}\n\n"
        "上記をもとに歌詞を書いてください。"
    )
    return _stream_response(WRITER_SYSTEM, prompt)


def review_agent(user_request: str, theme_brief: str, draft_lyrics: str) -> str:
    """歌詞をレビュー・改善する。"""
    print("\n🔍 [レビューエージェント] 歌詞をレビュー中...\n")
    prompt = (
        f"ユーザーの元リクエスト:\n{user_request}\n\n"
        f"テーマ資料:\n{theme_brief}\n\n"
        f"作詞エージェントの歌詞草稿:\n{draft_lyrics}\n\n"
        "この歌詞を批評し、改善版を提供してください。"
    )
    return _stream_response(REVIEW_SYSTEM, prompt)


# ---------------------------------------------------------------------------
# オーケストレーター
# ---------------------------------------------------------------------------

def create_lyrics(user_request: str) -> dict:
    """
    3エージェントを順番に実行し、最終歌詞を返す。

    Returns:
        {
            "theme": str,      # テーマ資料
            "draft": str,      # 初稿歌詞
            "final": str,      # レビュー済み歌詞
        }
    """
    print("=" * 60)
    print("🎼 作詞チームエージェント — rin.music")
    print("=" * 60)

    # Step 1: テーマ展開
    theme_brief = theme_agent(user_request)

    # Step 2: 作詞
    draft_lyrics = lyric_writer_agent(user_request, theme_brief)

    # Step 3: レビュー & 改善
    final_review = review_agent(user_request, theme_brief, draft_lyrics)

    print("\n" + "=" * 60)
    print("✅ 完成！")
    print("=" * 60)

    return {
        "theme": theme_brief,
        "draft": draft_lyrics,
        "final": final_review,
    }


# ---------------------------------------------------------------------------
# CLI エントリーポイント
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        request = " ".join(sys.argv[1:])
    else:
        print("作詞したい曲のイメージを入力してください。")
        print("例: 夏の終わりの切ない恋愛、J-POPバラード、前向きな気持ち")
        print()
        request = input("リクエスト > ").strip()
        if not request:
            request = "夏の終わり、遠距離恋愛、会いたい気持ち"

    result = create_lyrics(request)
