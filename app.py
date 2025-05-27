import streamlit as st
from rag_pipeline import retrieve_and_generate

st.title("Semantic Quote Retrieval System")

query = st.text_input("Ask me something:")
if query:
    with st.spinner("Retrieving relevant quotes..."):
        answer, sources = retrieve_and_generate(query)
        st.subheader("Answer")
        st.write(answer)
        st.subheader("Retrieved Quotes")
        for i, quote in enumerate(sources):
            st.markdown(f"**{i+1}.** {quote}")
