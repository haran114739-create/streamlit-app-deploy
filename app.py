import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(page_title="Streamlit LLM アプリ (LangChain v0.2対応)", layout="centered")
st.title("Streamlit + LangChain (0.2系対応) LLM アプリ")
st.write("テキストを入力し、選択した専門家としてLLMに応答させます。")
st.markdown(
    "- ローカル: プロジェクト直下に `.env` を作り `OPENAI_API_KEY=sk-...` を設定してください。\n"
    "- デプロイ: Streamlit Community Cloud の Secrets に `OPENAI_API_KEY` を登録してください。\n"
    "- 推奨 Python バージョン: 3.11"
)

EXPERT_OPTIONS = {
    "セキュリティエンジニア（A）": (
        "あなたは熟練のセキュリティエンジニアです。セキュリティ設計・脆弱性対応について、"
        "具体的かつ実務的に回答してください。必要ならコマンド例や設定例を示してください。"
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

def _safe_text(x) -> str:
    try:
        return str(x)
    except Exception:
        return repr(x)

def query_llm(input_text: str, role_key: str) -> str:
    """
    LangChain v0.2 系の複数インターフェースに対応する実装。
    1) 新しい chat_models.ChatOpenAI（存在すれば）を使う
    2) 旧来の langchain.OpenAI + PromptTemplate + LLMChain（0.2系やそれに近い構成）を試す
    3) 上記どれも使えなければ OpenAI の公式クライアントにフォールバック
    """
    system_msg = EXPERT_OPTIONS.get(role_key, "")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "環境変数 OPENAI_API_KEY が設定されていません。`.env` またはデプロイ先のシークレットを確認してください。"

    # 1) chat_models.ChatOpenAI（新しい/一部の構成）
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
        return _safe_text(res).strip()
    except Exception:
        pass

    # 2) 旧来の LangChain インターフェース（0.2 系で見られる形式）
    try:
        # 例: from langchain import OpenAI, LLMChain, PromptTemplate
        # 一部の 0.2 系ではこの import 構成が使われます
        try:
            from langchain import OpenAI, LLMChain, PromptTemplate  # type: ignore
        except Exception:
            # 別のモジュール配置の可能性に備える
            from langchain.llms import OpenAI  # type: ignore
            from langchain.chains import LLMChain  # type: ignore
            from langchain.prompts import PromptTemplate  # type: ignore

        # PromptTemplate はテンプレート変数で system と user_input を受け取るようにする
        template = "System: {system}\n\nUser: {user_input}\n\nAssistant:"
        prompt = PromptTemplate(template=template, input_variables=["system", "user_input"])
        llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
        chain = LLMChain(llm=llm, prompt=prompt)
        res = chain.run({"system": system_msg, "user_input": input_text})
        return _safe_text(res).strip()
    except Exception:
        pass

    # 3) フォールバック: OpenAI の公式クライアント（openai>=1.0）を直接呼ぶ
    try:
        from openai import OpenAI as OpenAIClient  # type: ignore
    except Exception as e:
        return f"LangChain と OpenAI クライアントの両方が利用できません: {_safe_text(e)}"

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
        # 応答抽出（複数パターンに対応）
        try:
            return resp.choices[0].message['content'].strip()
        except Exception:
            try:
                return resp['choices'][0]['message']['content'].strip()
            except Exception:
                try:
                    return resp.choices[0].text.strip()
                except Exception:
                    return _safe_text(resp)
    except Exception as e:
        return f"OpenAI API 呼び出しでエラーが発生しました: {_safe_text(e)}"

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
