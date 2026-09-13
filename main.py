import os
import json
import itertools
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from groq import AsyncGroq

app = FastAPI(title="Коч.ai")

# 1. Считываем ключи и настраиваем ротатор
raw_keys = os.environ.get("GROQ_API_KEY", "")
API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

if not API_KEYS:
    print("ВНИМАНИЕ: Переменная GROQ_API_KEY не найдена или пуста!")
    API_KEYS = ["dummy_key"]

key_cycle = itertools.cycle(API_KEYS)

def get_groq_client() -> AsyncGroq:
    """Возвращает AsyncGroq с новым ключом из списка"""
    key = next(key_cycle)
    return AsyncGroq(api_key=key)

# Рабочие и проверенные модели Groq
MODEL_GEN = "llama-3.3-70b-versatile"
MODEL_CRITIC = "qwen-2.5-32b"
MODEL_FINAL = "llama-3.1-8b-instant"

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Коч.ai</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: #ffffff; color: #000000; display: flex; justify-content: center; min-height: 100vh; padding: 20px; }
        .container { width: 100%; max-width: 800px; display: flex; flex-direction: column; gap: 20px; }
        header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 12px; border-bottom: 2px solid #000000; }
        h1 { font-size: 22px; font-weight: 700; }
        .badge { font-size: 11px; font-weight: 600; border: 1px solid #000000; padding: 3px 8px; border-radius: 999px; text-transform: uppercase; }
        .chat-history { display: flex; flex-direction: column; gap: 16px; min-height: 200px; }
        .message { display: flex; flex-direction: column; gap: 6px; }
        .message.user { align-items: flex-end; }
        .message.user .bubble { background-color: #000000; color: #ffffff; border-radius: 18px 18px 2px 18px; }
        .message.assistant .bubble { background-color: #ffffff; color: #000000; border: 1.5px solid #000000; border-radius: 18px 18px 18px 2px; }
        .bubble { max-width: 85%; padding: 12px 16px; font-size: 14px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
        .agents-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }
        @media (max-width: 600px) { .agents-grid { grid-template-columns: 1fr; } }
        .agent-card { border: 1px solid #000000; border-radius: 14px; padding: 12px; background-color: #fafafa; }
        .agent-title { font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 6px; border-bottom: 1px solid #e0e0e0; padding-bottom: 4px; }
        .agent-text { font-size: 13px; color: #333333; white-space: pre-wrap; min-height: 40px; }
        .input-area { display: flex; gap: 10px; position: sticky; bottom: 20px; background-color: #ffffff; padding-top: 10px; }
        input[type="text"] { flex: 1; border: 1.5px solid #000000; border-radius: 14px; padding: 12px 16px; font-size: 14px; outline: none; }
        button { background-color: #000000; color: #ffffff; border: none; border-radius: 14px; padding: 0 20px; font-size: 14px; font-weight: 600; cursor: pointer; }
        button:disabled { background-color: #888888; cursor: not-allowed; }
        .error-text { color: #d32f2f; font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Коч.ai</h1>
            <span class="badge">Multi-Agent System</span>
        </header>
        <div id="chat" class="chat-history"></div>
        <div class="input-area">
            <input id="query-input" type="text" placeholder="Задайте вопрос..." onkeypress="handleKey(event)">
            <button id="send-btn" onclick="sendQuery()">Отправить</button>
        </div>
    </div>
    <script>
        let ws;
        let contextHistory = [];
        let currentTurn = null;

        function connectWS() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'agent_start') createTurnUI();
                else if (data.type === 'agent_stream') appendAgentText(data.agent, data.text);
                else if (data.type === 'final_stream') appendFinalText(data.text);
                else if (data.type === 'error') showError(data.message);
                else if (data.type === 'complete') finishTurn(data.final_text);
            };

            ws.onclose = () => setTimeout(connectWS, 2000);
        }
        connectWS();

        function handleKey(e) { if (e.key === 'Enter') sendQuery(); }

        function sendQuery() {
            const input = document.getElementById('query-input');
            const btn = document.getElementById('send-btn');
            const text = input.value.trim();
            if (!text || btn.disabled) return;

            document.getElementById('chat').innerHTML += `<div class="message user"><div class="bubble">${escapeHtml(text)}</div></div>`;
            contextHistory.push({"role": "user", "content": text});

            ws.send(JSON.stringify({ query: text, history: contextHistory }));

            input.value = '';
            input.disabled = true;
            btn.disabled = true;
            window.scrollTo(0, document.body.scrollHeight);
        }

        function createTurnUI() {
            const chat = document.getElementById('chat');
            const turnId = 'turn-' + Date.now();
            chat.innerHTML += `
                <div id="${turnId}" class="message assistant">
                    <div class="agents-grid">
                        <div class="agent-card">
                            <div class="agent-title">1. Аналитик</div>
                            <div class="agent-text" id="${turnId}-gen">Печатает...</div>
                        </div>
                        <div class="agent-card">
                            <div class="agent-title">2. Критик</div>
                            <div class="agent-text" id="${turnId}-critic">В ожидании...</div>
                        </div>
                    </div>
                    <div style="margin-top: 10px;" class="bubble" id="${turnId}-final">Синтез ответа...</div>
                </div>
            `;
            currentTurn = {
                genElem: document.getElementById(`${turnId}-gen`),
                criticElem: document.getElementById(`${turnId}-critic`),
                finalElem: document.getElementById(`${turnId}-final`)
            };
            currentTurn.genElem.innerText = '';
            currentTurn.finalElem.innerText = '';
        }

        function appendAgentText(agent, text) {
            if (!currentTurn) return;
            if (agent === 'generator') currentTurn.genElem.innerText += text;
            else if (agent === 'critic') {
                if (currentTurn.criticElem.innerText === 'В ожидании...') currentTurn.criticElem.innerText = '';
                currentTurn.criticElem.innerText += text;
            }
        }

        function appendFinalText(text) {
            if (!currentTurn) return;
            currentTurn.finalElem.innerText += text;
            window.scrollTo(0, document.body.scrollHeight);
        }

        function showError(msg) {
            if (currentTurn) currentTurn.finalElem.innerHTML = `<span class="error-text">Ошибка: ${escapeHtml(msg)}</span>`;
            document.getElementById('query-input').disabled = false;
            document.getElementById('send-btn').disabled = false;
        }

        function finishTurn(finalText) {
            contextHistory.push({"role": "assistant", "content": finalText});
            document.getElementById('query-input').disabled = false;
            document.getElementById('send-btn').disabled = false;
            document.getElementById('query-input').focus();
            currentTurn = null;
        }

        function escapeHtml(str) { return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data_raw = await websocket.receive_text()
            payload = json.loads(data_raw)
            user_query = payload.get("query", "")
            history = payload.get("history", [])

            await websocket.send_json({"type": "agent_start"})

            try:
                # --- 1. ШАГ: Генератор ---
                gen_client = get_groq_client()
                gen_messages = [{"role": "system", "content": "Дай развернутый, глубокий ответ."}] + history
                gen_stream = await gen_client.chat.completions.create(
                    model=MODEL_GEN,
                    messages=gen_messages,
                    stream=True
                )
                
                draft_text = ""
                async for chunk in gen_stream:
                    content = chunk.choices[0].delta.content or ""
                    draft_text += content
                    await websocket.send_json({"type": "agent_stream", "agent": "generator", "text": content})

                # --- 2. ШАГ: Критик ---
                critic_client = get_groq_client()
                critic_messages = [
                    {"role": "system", "content": "Найди логические ошибки и слабости в отчете."},
                    {"role": "user", "content": f"Запрос: {user_query}\nЧерновик: {draft_text}"}
                ]
                critic_stream = await critic_client.chat.completions.create(
                    model=MODEL_CRITIC,
                    messages=critic_messages,
                    stream=True
                )

                critique_text = ""
                async for chunk in critic_stream:
                    content = chunk.choices[0].delta.content or ""
                    critique_text += content
                    await websocket.send_json({"type": "agent_stream", "agent": "critic", "text": content})

                # --- 3. ШАГ: Финализатор ---
                final_client = get_groq_client()
                final_messages = history + [
                    {"role": "system", "content": "Напиши идеальный итоговый ответ с учетом замечаний."},
                    {"role": "user", "content": f"Черновик: {draft_text}\nКритика: {critique_text}"}
                ]
                final_stream = await final_client.chat.completions.create(
                    model=MODEL_FINAL,
                    messages=final_messages,
                    stream=True
                )

                final_text = ""
                async for chunk in final_stream:
                    content = chunk.choices[0].delta.content or ""
                    final_text += content
                    await websocket.send_json({"type": "final_stream", "text": content})

                await websocket.send_json({"type": "complete", "final_text": final_text})

            except Exception as e:
                error_msg = str(e)
                print("СБОЙ ПРИ ВЫЗОВЕ GROQ API:")
                traceback.print_exc()
                await websocket.send_json({"type": "error", "message": error_msg})

    except WebSocketDisconnect:
        pass
