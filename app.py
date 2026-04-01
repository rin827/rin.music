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
    _visible_response,
    _parse_action,
    create_lyrics,
    YASU_CHAT_SYSTEM,
    client,
    MODEL,
)


# ---------------------------------------------------------------------------
# yasuストリーミング
# ---------------------------------------------------------------------------

def _stream_yasu(messages: list):
    with client.messages.stream(
        model=MODEL,
        max_tokens=1024,
        system=YASU_CHAT_SYSTEM,
        messages=messages,
    ) as stream:
        for event in stream:
            if (
                event.type == "content_block_delta"
                and event.delta.type == "text_delta"
            ):
                yield event.delta.text


def _get_greeting() -> tuple[list, list]:
    """起動時の挨拶。"""
    greeting = "".join(_stream_yasu([
        {"role": "user", "content": "セッション開始。yasuとして軽く挨拶して。短めに。"},
    ]))
    chat = [(None, greeting)]
    msgs = [
        {"role": "user",      "content": "セッション開始"},
        {"role": "assistant", "content": greeting},
    ]
    return chat, msgs


# ---------------------------------------------------------------------------
# チャット応答
# ---------------------------------------------------------------------------

def respond(user_input: str, chat_history: list, messages: list):
    if not user_input.strip():
        yield "", chat_history, messages
        return

    # ── 保存 ──
    if user_input == SAVE_KEYWORD:
        path = save_session(messages)
        reply = f"💾 会話を保存しました！\n\n`{path}`\n\nまたね～ マイクオフ！🎤"
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
        for chunk in _stream_yasu(resume_prompt):
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

    for chunk in _stream_yasu(new_msgs):
        response += chunk
        new_chat[-1] = (user_input, response)
        yield "", new_chat, messages

    action  = _parse_action(response)
    visible = _visible_response(response)
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
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="rin.music") as demo:

    gr.Markdown("# 🎼 rin.music")
    gr.Markdown(
        f"👔 **yasu** と話して歌詞を作ろう"
        f"&emsp;｜&emsp;"
        f"`{SAVE_KEYWORD}` で保存&emsp;`{LOAD_KEYWORD}` で続きから"
    )

    chatbot = gr.Chatbot(
        label="",
        show_label=False,
        height=500,
        render_markdown=True,
    )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="メッセージを入力...",
            label="",
            scale=5,
            autofocus=True,
        )
        send_btn = gr.Button("送信", scale=1, variant="primary", min_width=60)

    gr.Markdown(
        f"<small>💡 チームメンバー: 👔 yasu (CEO) ／ 🎵 龍姫（たつき）(テーマ) ／ ✍️ レイ (作詞) ／ 🔍 ルキ (レビュー)</small>",
        elem_id="footer",
    )

    state = gr.State([])

    msg_box.submit(respond, [msg_box, chatbot, state], [msg_box, chatbot, state])
    send_btn.click(respond,  [msg_box, chatbot, state], [msg_box, chatbot, state])

    demo.load(_get_greeting, outputs=[chatbot, state])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        theme=gr.themes.Soft(),
    )
