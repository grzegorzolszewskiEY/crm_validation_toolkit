"""
This module contains code to create reports on
a Streamlit UI. Still a demo.
"""

import streamlit as st
from crm_validator.ccf.ccf_validator import CCFValidator
from crm_validator.constants import PASSED, METRIC, REPORT

if __name__ == "__main__":
    ccf_validator = CCFValidator()

    # Set page title
    st.title("CCF Validator Demo")

    st.write("Demo page to show validator UI")
    st.write("Uses CCF assignment process test.")

    # Get required inputs
    # This will likely be changed to use a file upload system,
    # or removed entirely.
    m_b = st.number_input(
        "Enter initial number of facilities",
        min_value=0,
        step=1,
        format="%i"
    )
    m_ex = st.number_input(
        "Enter number of facilities excluded",
        min_value=0,
        step=1,
        format="%i"
    )
    N = st.number_input(
        "Enter number of facilities for validation",
        min_value=0,
        step=1,
        format="%i"
    )

    # Only run this if all inputs are provided
    if all([m_b, m_ex, N]):
        st.header("REPORT")

        # Perform the assignment process test
        validation_result = ccf_validator.assignment_process(
            m_ex=m_ex,
            m_b=m_b,
            N=N
        )

        st.markdown(f"PASSED : {validation_result[PASSED]}")

        # Write output
        st.subheader("Metrics calculated")
        st.write(validation_result[METRIC])

        st.subheader("Report")
        st.write(validation_result[REPORT])
