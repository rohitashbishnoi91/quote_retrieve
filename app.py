import streamlit as st
from rag_pipeline import retrieve_and_generate

st.set_page_config(page_title="Semantic Quote Retrieval", page_icon="💬")
st.title("💬 Semantic Quote Retrieval System")
st.markdown(
    """
    Welcome to the Semantic Quote Retrieval System!  
    Enter a natural language query (e.g., *"Quotes about hope by Oscar Wilde"*) and get relevant, context-aware quotes powered by RAG.
    """
)

query = st.text_input("🔎 Ask me something:")
if query:
    with st.spinner("Retrieving relevant quotes..."):
        answer, sources = retrieve_and_generate(query)
        st.markdown("### 📝 Answer")
        st.success(answer)
        st.markdown("### 📚 Retrieved Quotes")
        for i, quote in enumerate(sources):
            st.markdown(
                f"""
                <div style="background-color:#f0f2f6;padding:10px;border-radius:8px;margin-bottom:8px;">
                <b>{i+1}.</b> {quote}
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.info("Type your query above to get started!")