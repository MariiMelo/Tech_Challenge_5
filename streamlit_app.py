import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(layout="wide")

# caminho do projeto
BASE_DIR = Path(__file__).resolve().parent
ART = BASE_DIR / "artifacts"

# carregar base principal
df_ml2 = pd.read_parquet(ART / "df_ml2.parquet")

# carregar modelo e arquivos auxiliares
model = joblib.load(ART / "modelo_match.pkl")
features_treino = joblib.load(ART / "features_match.pkl")
top_titulos = joblib.load(ART / "top_titulos.pkl")
top_titulos2 = joblib.load(ART / "top_titulos2.pkl")

# padronizar tipos das chaves pra evitar erro de merge
df_ml2["codigo"] = df_ml2["codigo"].astype(str)
df_ml2["id_vaga"] = df_ml2["id_vaga"].astype(str)

# garantir colunas básicas
if "modalidade" not in df_ml2.columns:
    df_ml2["modalidade"] = "Desconhecido"

if "titulo" not in df_ml2.columns:
    df_ml2["titulo"] = "Desconhecido"

# função simples pra pegar valor mais frequente
def moda_ou_desconhecido(s):
    m = s.dropna().mode()
    return m.iloc[0] if len(m) > 0 else "Desconhecido"

# calcular features por candidato só uma vez
if "cand_feat" not in st.session_state or "contatos" not in st.session_state:

    st.session_state["cand_feat"] = (
        df_ml2.groupby("codigo", as_index=False)
        .agg(
            qtd_aplicacoes=("id_vaga", "count"),
            qtd_vagas_distintas=("id_vaga", "nunique"),
            modalidade_mais_freq=("modalidade", moda_ou_desconhecido),
            titulo_mais_freq=("titulo", moda_ou_desconhecido),
        )
    )

    st.session_state["contatos"] = (
        df_ml2[["codigo", "cand_nome", "cand_email", "cand_telefone"]]
        .replace("", np.nan)
        .dropna(subset=["codigo"])
        .groupby("codigo", as_index=False)
        .first()
    )

cand_feat = st.session_state["cand_feat"]
contatos = st.session_state["contatos"]

st.title("Ranking de candidatos por vaga")

# lista de vagas disponíveis
vagas_disp = (
    df_ml2[["titulo"]]
    .drop_duplicates()
    .dropna(subset=["titulo"])
)

titulo_vaga = st.selectbox(
    "Selecione o título da vaga",
    sorted(vagas_disp["titulo"].unique().tolist())
)

st.write("Vaga selecionada:", titulo_vaga)

# inicializar ranking na sessão
if "df_rank" not in st.session_state:
    st.session_state["df_rank"] = None
    st.session_state["ultima_vaga"] = None

# botão principal
if st.button("Gerar ranking de candidatos"):

    # só recalcula se trocar a vaga
    if st.session_state["ultima_vaga"] != titulo_vaga:

        with st.spinner("Gerando ranking..."):
            df_rank = cand_feat.copy()

            # filtro simples usando palavra-chave do título
            titulo_txt = str(titulo_vaga).lower().strip()
            palavras = titulo_txt.replace("(", " ").replace(")", " ").replace("-", " ").split()
            palavras = [p for p in palavras if len(p) >= 3]
            palavra_chave = palavras[0] if len(palavras) > 0 else ""

            if palavra_chave:
                mask = df_rank["titulo_mais_freq"].astype(str).str.lower().str.contains(palavra_chave, na=False)
                if mask.sum() >= 50:
                    df_rank = df_rank[mask].copy()

            df_rank["titulo"] = titulo_vaga
            df_rank["modalidade"] = "Desconhecido"

            df_rank = df_rank.merge(contatos, on="codigo", how="left")

            # montar dataset para o modelo
            X = df_rank[
                [
                    "modalidade",
                    "titulo",
                    "qtd_aplicacoes",
                    "qtd_vagas_distintas",
                    "modalidade_mais_freq",
                    "titulo_mais_freq",
                ]
            ].copy()

            X["qtd_aplicacoes"] = pd.to_numeric(X["qtd_aplicacoes"], errors="coerce").fillna(0)
            X["qtd_vagas_distintas"] = pd.to_numeric(X["qtd_vagas_distintas"], errors="coerce").fillna(0)
            X = X.fillna("Desconhecido")

            # reduzir cardinalidade dos títulos
            X["titulo"] = X["titulo"].where(X["titulo"].isin(top_titulos), "Outros")
            X["titulo_mais_freq"] = X["titulo_mais_freq"].where(
                X["titulo_mais_freq"].isin(top_titulos2), "Outros"
            )

            # one-hot encoding
            X_enc = pd.get_dummies(X, drop_first=True)
            X_enc = X_enc.reindex(columns=features_treino, fill_value=0)

            # score do modelo
            df_rank["score"] = model.predict_proba(X_enc)[:, 1]
            df_rank["score_percentual"] = (df_rank["score"] * 100).round(2)

            df_rank = df_rank.sort_values("score", ascending=False)

            st.session_state["df_rank"] = df_rank
            st.session_state["ultima_vaga"] = titulo_vaga

# exibir resultados
if st.session_state["df_rank"] is not None:

    df_rank = st.session_state["df_rank"]

    st.subheader("Ranking de candidatos")

    st.dataframe(
        df_rank[
            [
                "codigo",
                "cand_nome",
                "score_percentual",
                "qtd_aplicacoes",
                "qtd_vagas_distintas",
            ]
        ].head(30),
        use_container_width=True,
    )

    st.subheader("Detalhes do candidato")

    top30 = df_rank.head(30).copy()
    top30["opcao"] = top30["cand_nome"].fillna("Sem nome") + " (" + top30["codigo"] + ")"

    opcao_sel = st.selectbox(
        "Selecione um candidato pelo nome",
        top30["opcao"].tolist(),
    )

    codigo_sel = opcao_sel.split("(")[-1].replace(")", "").strip()
    cand = df_rank[df_rank["codigo"] == codigo_sel].iloc[0]

    st.write("Nome:", cand.get("cand_nome"))
    st.write("Email:", cand.get("cand_email"))
    st.write("Telefone:", cand.get("cand_telefone"))
    st.write("Score (%):", cand.get("score_percentual"))
    st.write("Quantidade de aplicações:", cand.get("qtd_aplicacoes"))
    st.write("Quantidade de vagas distintas:", cand.get("qtd_vagas_distintas"))
    st.write("Modalidade mais frequente:", cand.get("modalidade_mais_freq"))
    st.write("Título mais frequente:", cand.get("titulo_mais_freq"))
