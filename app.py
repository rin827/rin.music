"""
rin.music — ウェブアプリ

起動:
  python app.py

アクセス:
  PC     : http://localhost:7860
  スマホ  : 起動時に表示される公開URL (例: https://xxxx.gradio.live)
"""

import gradio as gr
from lyrics_team import (
    SAVE_KEYWORD,
    LOAD_KEYWORD,
    save_session,
    load_session,
    _visible_response as _lyrics_visible_response,
    _parse_action as _lyrics_parse_action,
    create_lyrics,
    YASU_CHAT_SYSTEM,
    client,
    MODEL,
)
from music_team import (
    save_music_session,
    load_music_session,
    _visible_response as _music_visible_response,
    _parse_action as _music_parse_action,
    create_music,
    YASU_MUSIC_CHAT_SYSTEM,
)


# ---------------------------------------------------------------------------
# 共通ストリーミングヘルパー
# ---------------------------------------------------------------------------

def _stream_with_system(system: str, messages: list):
    with client.messages.stream(
        model=MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
    ) as stream:
        for event in stream:
            if (
                event.type == "content_block_delta"
                and event.delta.type == "text_delta"
            ):
                yield event.delta.text


# ---------------------------------------------------------------------------
# 【歌詞タブ】yasuストリーミング
# ---------------------------------------------------------------------------

def _stream_yasu_lyrics(messages: list):
    yield from _stream_with_system(YASU_CHAT_SYSTEM, messages)


def _get_lyrics_greeting() -> tuple[list, list]:
    greeting = "".join(_stream_yasu_lyrics([
        {"role": "user", "content": "セッション開始。yasuとして軽く挨拶して。短めに。"},
    ]))
    chat = [(None, greeting)]
    msgs = [
        {"role": "user",      "content": "セッション開始"},
        {"role": "assistant", "content": greeting},
    ]
    return chat, msgs


def respond_lyrics(user_input: str, chat_history: list, messages: list):
    if not user_input.strip():
        yield "", chat_history, messages
        return

    # ── 保存 ──
    if user_input == SAVE_KEYWORD:
        path = save_session(messages)
        reply = f"💾 会話を保存しました！\n\n`{path}`\n\nまたね～ 👋"
        yield "", chat_history + [(user_input, reply)], messages
        return

    # ── 読み込み ──
    if user_input == LOAD_KEYWORD:
        loaded = load_session()
        if not loaded:
            yield "", chat_history + [(user_input, "📂 保存された会話が見つかりません。")], messages
            return

        skip = {"セッション開始", LOAD_KEYWORD, "[作詞完了の報告]"}
        new_chat = [
            (m["content"], loaded[i + 1]["content"] if i + 1 < len(loaded) else "")
            for i, m in enumerate(loaded)
            if m["role"] == "user" and m["content"] not in skip and i % 2 == 0
        ]

        resume_prompt = loaded + [{"role": "user", "content":
            "前回の会話の続きです。前回どんな話をしていたか一言で振り返ってから続けて。"}]
        new_chat.append((LOAD_KEYWORD, ""))
        resume_resp = ""
        for chunk in _stream_yasu_lyrics(resume_prompt):
            resume_resp += chunk
            new_chat[-1] = (LOAD_KEYWORD, resume_resp)
            yield "", new_chat, loaded

        updated = loaded + [
            {"role": "user",      "content": LOAD_KEYWORD},
            {"role": "assistant", "content": resume_resp},
        ]
        yield "", new_chat, updated
        return

    # ── 通常会話（ストリーミング）──
    new_msgs  = messages + [{"role": "user", "content": user_input}]
    new_chat  = chat_history + [(user_input, "")]
    response  = ""

    for chunk in _stream_yasu_lyrics(new_msgs):
        response += chunk
        new_chat[-1] = (user_input, response)
        yield "", new_chat, messages

    action  = _lyrics_parse_action(response)
    visible = _lyrics_visible_response(response)
    new_chat[-1] = (user_input, visible)
    new_msgs = new_msgs + [{"role": "assistant", "content": visible}]

    # ── 作詞パイプライン ──
    if action:
        new_chat = new_chat + [(None, "🎵 **龍姫・レイ・ルキが作詞を開始します！**\n\nしばらくお待ちください...")]
        yield "", new_chat, new_msgs

        result = create_lyrics(action)

        lyrics_reply = (
            f"✅ **歌詞が完成しました！**\n\n"
            f"**タイトル:** {result['title']}\n"
            f"**ジャンル:** {result['genre']}\n"
            f"**保存先:** `{result['saved_to']}`\n\n---\n\n"
            f"{result['final']}"
        )
        new_chat[-1] = (None, lyrics_reply)
        summary = f"[作詞完了] {result['title']} / {result['genre']} / {result['saved_to']}"
        new_msgs = new_msgs + [
            {"role": "user",      "content": "[作詞完了の報告]"},
            {"role": "assistant", "content": summary},
        ]
        yield "", new_chat, new_msgs
        return

    yield "", new_chat, new_msgs


# ---------------------------------------------------------------------------
# 【音楽制作タブ】yasuストリーミング
# ---------------------------------------------------------------------------

def _stream_yasu_music(messages: list):
    yield from _stream_with_system(YASU_MUSIC_CHAT_SYSTEM, messages)


def _get_music_greeting() -> tuple[list, list]:
    greeting = "".join(_stream_yasu_music([
        {"role": "user", "content": "音楽制作セッション開始。yasuとして音楽制作の話をしようと軽く挨拶して。短めに。"},
    ]))
    chat = [(None, greeting)]
    msgs = [
        {"role": "user",      "content": "セッション開始"},
        {"role": "assistant", "content": greeting},
    ]
    return chat, msgs


def respond_music(user_input: str, chat_history: list, messages: list):
    if not user_input.strip():
        yield "", chat_history, messages
        return

    # ── 保存 ──
    if user_input == SAVE_KEYWORD:
        path = save_music_session(messages)
        reply = f"💾 会話を保存しました！\n\n`{path}`\n\nまたね～ 👋"
        yield "", chat_history + [(user_input, reply)], messages
        return

    # ── 読み込み ──
    if user_input == LOAD_KEYWORD:
        loaded = load_music_session()
        if not loaded:
            yield "", chat_history + [(user_input, "📂 保存された会話が見つかりません。")], messages
            return

        skip = {"セッション開始", LOAD_KEYWORD, "[音楽制作完了の報告]"}
        new_chat = [
            (m["content"], loaded[i + 1]["content"] if i + 1 < len(loaded) else "")
            for i, m in enumerate(loaded)
            if m["role"] == "user" and m["content"] not in skip and i % 2 == 0
        ]

        resume_prompt = loaded + [{"role": "user", "content":
            "前回の会話の続きです。前回どんな話をしていたか一言で振り返ってから続けて。"}]
        new_chat.append((LOAD_KEYWORD, ""))
        resume_resp = ""
        for chunk in _stream_yasu_music(resume_prompt):
            resume_resp += chunk
            new_chat[-1] = (LOAD_KEYWORD, resume_resp)
            yield "", new_chat, loaded

        updated = loaded + [
            {"role": "user",      "content": LOAD_KEYWORD},
            {"role": "assistant", "content": resume_resp},
        ]
        yield "", new_chat, updated
        return

    # ── 通常会話（ストリーミング）──
    new_msgs  = messages + [{"role": "user", "content": user_input}]
    new_chat  = chat_history + [(user_input, "")]
    response  = ""

    for chunk in _stream_yasu_music(new_msgs):
        response += chunk
        new_chat[-1] = (user_input, response)
        yield "", new_chat, messages

    action  = _music_parse_action(response)
    visible = _music_visible_response(response)
    new_chat[-1] = (user_input, visible)
    new_msgs = new_msgs + [{"role": "assistant", "content": visible}]

    # ── 音楽制作パイプライン ──
    if action:
        new_chat = new_chat + [(None, "🎸 **蒼牙・澪・颯・凛が制作を開始します！**\n\nしばらくお待ちください...")]
        yield "", new_chat, new_msgs

        result = create_music(action)

        music_reply = (
            f"✅ **音楽制作ガイドが完成しました！**\n\n"
            f"**タイトル:** {result['title']}\n"
            f"**ジャンル:** {result['genre']}\n"
            f"**キー:** {result['key']}  |  **BPM:** {result['bpm']}\n"
            f"**保存先:** `{result['saved_to']}`\n\n---\n\n"
            f"{result['production']}"
        )
        new_chat[-1] = (None, music_reply)
        summary = (
            f"[音楽制作完了] {result['title']} / {result['genre']} / "
            f"Key: {result['key']} / BPM: {result['bpm']} / {result['saved_to']}"
        )
        new_msgs = new_msgs + [
            {"role": "user",      "content": "[音楽制作完了の報告]"},
            {"role": "assistant", "content": summary},
        ]
        yield "", new_chat, new_msgs
        return

    yield "", new_chat, new_msgs


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="rin.music", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🎼 rin.music")

    with gr.Tabs():

        # ────────────────────────────────────────────────────────
        # Tab 1: 作詞チーム
        # ────────────────────────────────────────────────────────
        with gr.Tab("✍️ 作詞チーム"):
            gr.Markdown(
                f"👔 **yasu** と話して歌詞を作ろう"
                f"&emsp;｜&emsp;"
                f"`{SAVE_KEYWORD}` で保存&emsp;`{LOAD_KEYWORD}` で続きから"
            )

            lyrics_chatbot = gr.Chatbot(
                label="",
                show_label=False,
                height=460,
                bubble_full_width=False,
                render_markdown=True,
            )

            with gr.Row():
                lyrics_msg = gr.Textbox(
                    placeholder="メッセージを入力...",
                    label="",
                    scale=5,
                    autofocus=True,
                )
                lyrics_send = gr.Button("送信", scale=1, variant="primary", min_width=60)

            gr.Markdown(
                "<small>💡 チームメンバー: 👔 yasu (CEO) ／ 🎵 龍姫（たつき）(テーマ) ／ ✍️ レイ (作詞) ／ 🔍 ルキ (レビュー)</small>"
            )

            lyrics_state = gr.State([])

            lyrics_msg.submit(respond_lyrics, [lyrics_msg, lyrics_chatbot, lyrics_state], [lyrics_msg, lyrics_chatbot, lyrics_state])
            lyrics_send.click(respond_lyrics,  [lyrics_msg, lyrics_chatbot, lyrics_state], [lyrics_msg, lyrics_chatbot, lyrics_state])

            demo.load(_get_lyrics_greeting, outputs=[lyrics_chatbot, lyrics_state])

        # ────────────────────────────────────────────────────────
        # Tab 2: 音楽制作チーム
        # ────────────────────────────────────────────────────────
        with gr.Tab("🎸 音楽制作チーム"):
            gr.Markdown(
                f"👔 **yasu** と話してコード進行・アレンジを作ろう"
                f"&emsp;｜&emsp;"
                f"`{SAVE_KEYWORD}` で保存&emsp;`{LOAD_KEYWORD}` で続きから"
            )

            music_chatbot = gr.Chatbot(
                label="",
                show_label=False,
                height=460,
                bubble_full_width=False,
                render_markdown=True,
            )

            with gr.Row():
                music_msg = gr.Textbox(
                    placeholder="例: 切ない夜のシティポップを作りたい",
                    label="",
                    scale=5,
                )
                music_send = gr.Button("送信", scale=1, variant="primary", min_width=60)

            gr.Markdown(
                "<small>💡 チームメンバー: 👔 yasu (CEO) ／ 🎸 蒼牙（そうが）(サウンドディレクター) ／ 🎹 澪（みお）(コード進行) ／ 🥁 颯（はやて）(アレンジ) ／ 🎚️ 凛（りん）(サウンドデザイン)</small>"
            )

            music_state = gr.State([])

            music_msg.submit(respond_music, [music_msg, music_chatbot, music_state], [music_msg, music_chatbot, music_state])
            music_send.click(respond_music,  [music_msg, music_chatbot, music_state], [music_msg, music_chatbot, music_state])

            demo.load(_get_music_greeting, outputs=[music_chatbot, music_state])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
