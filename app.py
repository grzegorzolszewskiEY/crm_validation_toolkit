"""
This module contains code to create reports on
a Streamlit UI. Still a demo.
"""

import streamlit as st

from ccf_app import ccf_page
from LGD_validation_app import pd_lgd_app

# Get required inputs
#input_file = st.file_uploader("Choose a file")
uploaded_files = st.file_uploader('Upload files', accept_multiple_files= True)
for uploaded_file in uploaded_files:
    if "CCF" in uploaded_file:
        input_file = uploaded_file

ccf_page(st, input_file)
pd_lgd_app(st, uploaded_files)
 