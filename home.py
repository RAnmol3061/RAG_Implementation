import streamlit as st
import backend as rag

st.title("Recommendation System with RAG Pipeline")
st.subheader("Add a subheader")
st.divider()

with st.sidebar:
    st.header("Upload Context")
    user_text = st.text_area("Enter Knowledge here", height = 150)

    if st.button("Upload to MongoDB"):
        if user_text:
            with st.spinner("Processing"):
                rag.ingest_text(user_text)
                st.success("Uploaded")
        else:
            st.warning("Please enter text")

st.header("Ask anything to the chat from our knowledge base")


if "message" not in st.session_state:
    st.session_state.message = []

for message in st.session_state.message:
    with st.chat_message(message['role']):
        st.markdown(message.get("content", ""))

prompt = st.chat_input("Ask any question")
if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.message.append({
        "role":"user",
        "content":prompt
    })

    with st.chat_message('robot'):
        with st.spinner('Thinking'):
            response_data = rag.get_rag_response(prompt)
            answer = response_data['answer']
            sources = response_data['sources']

            st.markdown(answer)
            

            with st.expander("Sources"):
                for i,source in enumerate(sources):
                    st.markdown(f"**Source {i+1}:** {source.page_content}")
            
            st.session_state.message.append({
                "role":"robot",
                "content":answer,
                "source": [s.page_content for s in sources]
            })
