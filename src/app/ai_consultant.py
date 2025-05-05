import streamlit as st
import asyncio

def display_ai_consultant():

    rag_class = st.session_state.rag_class

    st.header("AI Consultant")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):  
            st.markdown(message["content"])

    # React to user input
    if prompt:= st.chat_input("Say Something"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            placeholder = st.empty()

            # Get both chunks and metadata
            relevant_chunks, results_metadata = rag_class.query_documents(question=prompt)

            with st.spinner("🍃 Generating Response..."):
                response = rag_class.generate_response(
                        question=prompt, 
                        relevant_chunks=relevant_chunks, 
                        results_metadata=results_metadata
                    )
                
            placeholder.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})