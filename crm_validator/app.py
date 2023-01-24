"""
This module contains code to create reports on
a Streamlit UI.
"""

import streamlit as st
from crm_validator.ccf_validator import CCFValidator

if __name__ == "__main__":
    ccf_validator = CCFValidator()

    # Set page title
    st.title("CCF Validator Demo")

    st.write("Page to validate CCF Model")

    # Get required inputs
    # This will likely be changed to use a file upload system,
    # or removed entirely.
    m_b = st.number_input("Enter initial number of facilities")
    m_ex = st.number_input("Enter number of facilities excluded")
    N = st.number_input("Enter number of facilities for validation")

    # Only run this if all inputs are provided
    if all([m_b, m_ex, N]):
        # Perform the assignment process test
        validation_result = ccf_validator.validate_assignment_process(
            m_ex=m_ex,
            m_b=m_b,
            N=N
        )

        # Write output
        st.write(validation_result)
