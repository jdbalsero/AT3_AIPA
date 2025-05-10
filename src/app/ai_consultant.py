import streamlit as st
import asyncio

def display_ai_consultant():

    rag_class = st.session_state.rag_class

    st.header("Mathiesen Group AI Assistant")


    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello. Welcome to the Mathiesen Group Company, a leading supplier of raw materials and industrial inputs for various sectors, including the paper and pulp industry. How can I assist you today? Are you looking for information on our products or services?"
    })
    
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

            with st.spinner("🍃 Generating Response..."):
                response = rag_class.generate_response(
                        question=prompt
                    )
                
            placeholder.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})