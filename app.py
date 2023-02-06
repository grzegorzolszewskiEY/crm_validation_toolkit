"""
This module contains code to create reports on
a Streamlit UI. Still a demo.
"""

import streamlit as st
import pandas as pd
from crm_validator.ccf.ccf_wrapper import CCFWrapper
# from crm_validator.constants import PASSED, METRIC, REPORT

if __name__ == "__main__":
    # Set page title
    st.title("CCF Validator Demo")

    st.write("Demo page to show validator UI")

    # Get required inputs
    input_file = st.file_uploader("Choose a file")

    if input_file is not None:
        # Only run this if input is provided
        ccf_data = pd.read_csv(
            input_file,
            delimiter=";",
            header=0
        )
        st.write("Data sample")
        st.write(ccf_data.head())

        ccf_tester = CCFWrapper(ccf_data)

        results = ccf_tester.run_validation_tests()
        for test_result in results:
            st.subheader(test_result["test"])
            st.write(test_result["report"])
