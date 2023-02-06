"""
This module contains code to create reports on
a Streamlit UI. Still a demo.
"""

import pandas as pd
import streamlit as st
from crm_validator.ccf.ccf_wrapper import CCFWrapper


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
            header=0
        )
        st.write("Data sample")
        st.write(ccf_data.head())

        ccf_tester = CCFWrapper(ccf_data)

        results = ccf_tester.run_validation_tests()
        for test_result in results:
            if type(test_result) == dict:
                st.header(test_result["test"])
                st.write(test_result["report"])
            else:
                st.header(test_result.name)
                st.markdown(
                    "Test result : **"
                    f"{'Passed' if test_result.passed else 'Failed'}"
                    "**"
                )

                st.subheader("Calculated metrics")
                metric_names = test_result.metrics.keys()
                n_metrics = len(metric_names)
                columns = st.columns(n_metrics)
                for i, metric in enumerate(metric_names):
                    columns[i].metric(
                        label=metric,
                        value=test_result.metrics[metric]
                    )

                st.subheader("Report")
                report_keys = test_result.reports.keys()
                report_values = test_result.reports.values()
                st.table(
                    {
                        "QUANTITY": report_keys,
                        "VALUE": report_values
                    }
                )

                if test_result.plots:
                    st.subheader("Plots")
                    for plot in test_result.plots:
                        st.pyplot(plot)
