import sys
import os

# Import packages
import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path

# import backend.rag_process

# Set Python path
current_dir = os.path.dirname(__file__)
parent_dir = str(Path(current_dir).resolve().parents[0])
sys.path.append(parent_dir)
from backend.rag_process import rag_process

rag_class = rag_process()

rag_class.run_embedding_process()
