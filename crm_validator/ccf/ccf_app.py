"""
This module contains code to create reports on
a Streamlit UI. Still a demo.
"""

import pandas as pd

from crm_validator.ccf.ccf_wrapper import CCFWrapper


def ccf_page(st, input_file):
    # Set page title
    st.title("CCF Validator")

    st.write("Demo page to show validator UI")

    if input_file is not None:
        input_file.seek(0)
        # Only run this if input is provided
        ccf_data = pd.read_csv(
            input_file,
            header=0
        )

        # Show sample of data
        st.write("Data sample")
        st.write(ccf_data.head())

        # Create an object of the validator class
        ccf_tester = CCFWrapper(ccf_data)

        # Run all validation tests
        test_results = ccf_tester.run_validation_tests()

        st.markdown("---")

        # Display reports of each test
        for report in test_results:
            # Create heading
            st.markdown(f"## {report.name}")
            # Display description if exists
            if report.description:
                st.write(report.description)

            # Iterate through subreports
            for subreport in report:
                # Add title as subheader if exists
                if subreport.name:
                    st.markdown(f"### {subreport.name}")
                # Display description if exists
                if subreport.description:
                    st.subheader(subreport.description)

                # Print result of report (pass/fail)
                st.markdown(
                    "Test result : **"
                    f"{'Passed' if subreport.passed else 'Failed'}"
                    "**"
                )

                # Section with metric calculations
                st.markdown("#### Calculated metrics")
                metric_names = subreport.metrics.keys()
                n_metrics = len(metric_names)
                columns = st.columns(n_metrics)
                for i, metric in enumerate(metric_names):
                    columns[i].metric(
                        label=metric,
                        value=subreport.metrics[metric]
                    )

                st.markdown("#### Report values")
                report_keys = subreport.reports.keys()
                report_values = subreport.reports.values()
                st.table(
                    {
                        "QUANTITY": report_keys,
                        "VALUE": report_values
                    }
                )

                if subreport.plots:
                    st.markdown("#### Plots")
                    for plot in subreport.plots:
                        st.pyplot(plot)

            st.markdown("---")
