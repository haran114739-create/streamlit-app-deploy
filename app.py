import os
from dotenv import load_dotenv

# ロケール／エンコーディングを確実に UTF-8 に（必要なら）
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# .env を読み込む
load_dotenv()

import streamlit as st

st.set_page_config(page_title="Streamlit LLM アプリ", layout="centered")

st.title("Streamlit + LangChain LLM アプリ")
st.write("入力したテキストを選んだ専門家として LLM に回答させます。")
st.markdown(
    "- ローカル実行: プロジェクト直下に `.env` を作り `OPENAI_API_KEY=sk-...` を設定してください。\n"
    "- デプロイ: Streamlit Community Cloud の Secrets に `OPENAI_API_KEY` を登録してください。\n"
    "- 推奨 Python バージョン: 3.11"
)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.warning("環境変数 OPENAI_API_KEY が見つかりません。`.env` またはデプロイ先のシークレットを確認してください。")

# 専門家（システムメッセージ）を定義（A/B 例）
EXPERT_OPTIONS = {
    "セキュリティエンジニア（A）": (
        "あなたは熟練のセキュリティエンジニアです。セキュリティ設計・脆弱性対応に関して、"
        "具体的かつ実務的にアドバイスしてください。必要ならコマンドや設定例を示してください。"
    ),
    "プロダクトマーケター（B）": (
        "あなたは経験豊富なプロダクトマーケターです。市場分析、ターゲティング、KPI設計、"
        "ローンチ施策について実務的に提案してください。"
    ),
}

with st.form("form"):
    role = st.radio("専門家の種類を選択してください:", list(EXPERT_OPTIONS.keys()))
    user_input = st.text_area("質問または指示を入力してください:", height=180)
    submit = st.form_submit_button("送信")

def _safe_str(x: object) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)
    finally:
        # 保険として UTF-8 にエンコード／デコード（非ASCII を置換）
        # ここは戻り値で必ず使うこと
        pass

def _safe_unicode(s: str) -> str:
    if s is None:
        return ""
    try:
        return s.encode("utf-8", errors="replace").decode("utf-8")
    except Exception:
        return repr(s)

def query_llm(input_text: str, role_key: str) -> str:
    system_msg = EXPERT_OPTIONS.get(role_key, "")

    # 1) LangChain
    try:
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import (
            ChatPromptTemplate,
            SystemMessagePromptTemplate,
            HumanMessagePromptTemplate,
        )
        from langchain.chains import LLMChain

        system_template = SystemMessagePromptTemplate.from_template(system_msg)
        human_template = HumanMessagePromptTemplate.from_template("{user_input}")
        chat_prompt = ChatPromptTemplate.from_messages([system_template, human_template])

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
        chain = LLMChain(llm=llm, prompt=chat_prompt)
        res = chain.run({"user_input": input_text})
        return _safe_unicode(_safe_str(res))
    except Exception:
        pass

    # 2) OpenAI new client
    try:
        from openai import OpenAI
    except Exception as e:
        return _safe_unicode(f"LangChain/OpenAI import error: {_safe_str(e)}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "環境変数 OPENAI_API_KEY が設定されていません。`.env` を確認してください。"

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

        # 応答抽出（安全に取り出して UTF-8 化）
        content = None
        try:
            content = resp.choices[0].message['content']
        except Exception:
            try:
                content = resp.choices[0].message.content
            except Exception:
                try:
                    content = resp['choices'][0]['message']['content']
                except Exception:
                    try:
                        content = resp.choices[0].text
                    except Exception:
                        content = _safe_str(resp)

        return _safe_unicode(_safe_str(content))
    except Exception as e:
        return _safe_unicode(f"OpenAI API 呼び出しでエラーが発生しました: {_safe_str(e)}")

# 送信後の表示
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
