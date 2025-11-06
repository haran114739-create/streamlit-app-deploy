import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(page_title="Streamlit LLM アプリ (LangChain 対応)", layout="centered")
st.title("Streamlit + LangChain LLM アプリ")
st.write("テキストを入力し、選択した専門家として LLM に応答させます。")
st.markdown(
    "- ローカル: プロジェクト直下に `.env` を作り `OPENAI_API_KEY=sk-...` を設定してください。\n"
    "- デプロイ: Streamlit Community Cloud の Secrets に `OPENAI_API_KEY` を登録してください。\n"
    "- 推奨 Python バージョン: 3.11"
)

EXPERT_OPTIONS = {
    "セキュリティエンジニア（A）": "あなたは熟練のセキュリティエンジニアです。具体的かつ実務的にアドバイスしてください。",
    "プロダクトマーケター（B）": "あなたは経験豊富なプロダクトマーケターです。実務的な施策を提案してください。",
}

with st.form("form"):
    role = st.radio("専門家の種類を選択してください:", list(EXPERT_OPTIONS.keys()))
    user_input = st.text_area("質問または指示を入力してください:", height=180)
    submit = st.form_submit_button("送信")

def _safe_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)

def query_llm(input_text: str, role_key: str) -> str:
    system_msg = EXPERT_OPTIONS.get(role_key, "")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "環境変数 OPENAI_API_KEY が設定されていません。`.env` またはデプロイ先のシークレットを確認してください。"

    # 1) まず LangChain の chat_models.ChatOpenAI 系を試す（複数パスに対応）
    try:
        try:
            from langchain.chat_models import ChatOpenAI  # new-ish layout
        except Exception:
            from langchain.chat_models.openai import ChatOpenAI  # alternative
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
        return _safe_str(res).strip()
    except Exception:
        pass

    # 2) LangChain の旧来インターフェース（OpenAI LLM / PromptTemplate）を試す
    try:
        try:
            from langchain import OpenAI as LCOpenAI  # some 0.2 installs expose this
        except Exception:
            from langchain.llms import OpenAI as LCOpenAI  # alternative
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain

        prompt = PromptTemplate(template="{system}\n\n{user_input}", input_variables=["system", "user_input"])
        llm = LCOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
        chain = LLMChain(llm=llm, prompt=prompt)
        res = chain.run({"system": system_msg, "user_input": input_text})
        return _safe_str(res).strip()
    except Exception:
        pass

    # 3) フォールバック: OpenAI 公式クライアント（openai>=1.0）
    try:
        from openai import OpenAI as OpenAIClient
    except Exception as e:
        return f"LangChain と OpenAI クライアントの両方が利用できません: {_safe_str(e)}"

    try:
        client = OpenAIClient(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": input_text},
            ],
            temperature=0.2,
        )
        # 応答抽出（複数構造に対応）
        try:
            return resp.choices[0].message['content'].strip()
        except Exception:
            try:
                return resp['choices'][0]['message']['content'].strip()
            except Exception:
                try:
                    return resp.choices[0].text.strip()
                except Exception:
                    return _safe_str(resp)
    except Exception as e:
        return f"OpenAI API 呼び出しでエラーが発生しました: {_safe_str(e)}"

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
