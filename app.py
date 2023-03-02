"""
This module contains code to create reports on
a Streamlit UI. Still a demo.
"""

import streamlit as st

from ccf_app import ccf_page


# Get required inputs
input_file = st.file_uploader("Choose a file")

ccf_page(st, input_file)
