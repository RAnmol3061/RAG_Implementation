import streamlit as st
import backend as rag

st.title("Recommendation System with RAG Pipeline")
st.subheader("Add a subheader")
st.divider()

with st.sidebar:
    st.header("Upload Context")
    user_text = st.text_area("Enter Knowledge here", height = 150)

    if st.button("Upload to MongoDB"):
        if user_text():
            with st.spinner("Processing"):
                rag.ingest_text(user_text)
                st.success("Uploaded")
        else:
            st.warning("Please enter text")