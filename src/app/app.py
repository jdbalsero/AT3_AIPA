# #Async Callbacks workaround

# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())

# Import packages
import streamlit as st
from streamlit_option_menu import option_menu
import sys
import os
from pathlib import Path
import subprocess

# Set Python path
current_dir = os.path.dirname(__file__)
parent_dir = str(Path(current_dir).resolve().parents[0])
sys.path.append(parent_dir)

from ai_consultant import display_ai_consultant
from company_form import display_company_form
from backend.rag_process import rag_process
from backend.rag_assistant import RAGAssistant

if "rag_class" not in st.session_state:
    st.session_state.rag_class = rag_process()

if "rag_assistant" not in st.session_state:
    st.session_state.rag_assistant = RAGAssistant()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.session_state["company_info"] = None

with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        # options=["Company Form", "AI Consultant"],
        options=["AI Consultant"],
        icons=["file-earmark-text", "chat"],
        default_index=0,
    )

# Display the selected page
if selected == "AI Consultant":
    display_ai_consultant()
elif selected == "Company Form":
    display_company_form()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("🧠 Powered by AIPA Group 4")
st.sidebar.markdown("🧑‍💻 Built with Streamlit")
