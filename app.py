"""
This module contains code to create reports on
a Streamlit UI. Still a demo.
"""

import streamlit as st

from crm_validator.ccf.ccf_app import ccf_page
from LGD_validation_app import pd_lgd_app

# Get required inputs
uploaded_files = None
input_file = None
uploaded_files = st.file_uploader('Upload files', accept_multiple_files= True)
for uploaded_file in uploaded_files:
    if "CCF" in uploaded_file.name.upper():
        uploaded_file.seek(0)
        input_file = uploaded_file
if uploaded_files is not None:
    pd_lgd_app(st, uploaded_files)
    with st.expander('CCF validation part'):
        ccf_page(st, input_file)
st.markdown("---")

 