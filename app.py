import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Streamlit LLM アプリ", layout="centered")

st.title("Streamlit + OpenAI LLM アプリ")
st.write("入力したテキストを選んだ専門家として LLM に回答させます。")
st.markdown(
    "- ローカル実行: プロジェクト直下に `.env` を作り `OPENAI_API_KEY=sk-...` を設定してください。\n"
    "- デプロイ: Streamlit Community Cloud の Secrets に `OPENAI_API_KEY` を登録してください。\n"
    "- 推奨 Python バージョン: 3.11"
)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.warning("環境変数 OPENAI_API_KEY が見つかりません。`.env` またはデプロイ先のシークレットを確認してください。")

EXPERT_OPTIONS = {
    "セキュリティエンジニア（A）": "あなたは熟練のセキュリティエンジニアです。具体的かつ実務的にアドバイスしてください。",
    "プロダクトマーケター（B）": "あなたは経験豊富なプロダクトマーケターです。実務的な施策を提案してください。",
}

with st.form("form"):
    role = st.radio("専門家の種類を選択してください:", list(EXPERT_OPTIONS.keys()))
    user_input = st.text_area("質問または指示を入力してください:", height=180)
    submit = st.form_submit_button("送信")

def query_llm(input_text: str, role_key: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "環境変数 OPENAI_API_KEY が設定されていません。`.env` またはデプロイ先のシークレットを確認してください。"

    system_msg = EXPERT_OPTIONS.get(role_key, "")

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": input_text},
            ],
            temperature=0.2,
        )
        # 応答抽出（安全に）
        try:
            return resp.choices[0].message['content'].strip()
        except Exception:
            try:
                return resp['choices'][0]['message']['content'].strip()
            except Exception:
                try:
                    return resp.choices[0].text.strip()
                except Exception:
                    return str(resp)
    except Exception as e:
        return f"OpenAI API 呼び出しでエラーが発生しました: {e}"

if submit:
    if not user_input or not user_input.strip():
        st.info("入力テキストを記入してください。")
    else:
        with st.spinner("LLM に問い合わせ中..."):
            answer = query_llm(user_input.strip(), role)
        st.subheader("入力内容")
        st.write(user_input)
        st.subheader("選択した専門家")
        st.write(role)
        st.subheader("LLM の回答")
        st.write(answer)
