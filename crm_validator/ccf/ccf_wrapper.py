"""
Module was created as a wrapper to preprocess data and handle
inputs to the validator, so multiple calculations of inputs to
validator are not required.
"""

import numpy as np
import pandas as pd
from crm_validator.ccf.ccf_validator import CCFValidator
from crm_validator.exceptions import ValidationError


class CCFWrapper:
    """
    A class that takes in data to preprocess and input to validator
    """
    def __init__(
        self,
        ccf_data: pd.DataFrame
    ) -> None:
        self.ccf_data = ccf_data
        self.validator = CCFValidator()
        pass

    # --------------------
    # BACK TESTING REPORTS
    # --------------------

    def validate_assignment_process_back_testing(self):
        """
        Validates the assignment process.

        Total number of facilities is length of dataset.
        Number of exclusions are marked in dataset as "outlier" or
            "process deficiency".
        Number of final facilities is marked as either "ead covered"
            or "ccf covered"
        """
        m_b = len(self.ccf_data)
        m_ex = len(
            self.ccf_data.loc[
                (self.ccf_data["marked"] == "outlier") |
                (self.ccf_data["marked"] == "process_deficiency")
            ]
        )
        N = len(
            self.ccf_data.loc[
                (self.ccf_data["marked"] == "ead_covered") |
                (self.ccf_data["marked"] == "ccf_covered")
            ]
        )

        # Get report
        report = self.validator.assignment_process_back_testing(
            m_b=m_b,
            m_ex=m_ex,
            N=N
        )

        # Add report name
        report.name = "CCF assignment process (back-testing)"

        return report

    def validate_ead_covered_facilities(self):
        """
        Validates EAD covered facilities from back-testing.

        EAD covered facilities are marked as "ead_covered"
        Number of final facilities is marked as either "ead covered"
            or "ccf covered"
        """
        N = len(
            self.ccf_data.loc[
                (self.ccf_data["marked"] == "ead_covered") |
                (self.ccf_data["marked"] == "ccf_covered")
            ]
        )
        m_ead = len(
            self.ccf_data.loc[
                (self.ccf_data["marked"] == "ead_covered")
            ]
        )

        report = self.validator.ead_covered_facilities(
            m_ead=m_ead,
            N=N
        )

        report.name = "EAD covered facilities"
        return report

    # ------------------------
    # PREDICTIVE ABILITY TESTS
    # ------------------------

    def validate_ccf_predictive_power(
        self,
        floor: float = None,
        test_level: float = 0.05
    ):
        """
        Tests the predictive power of the CCF model for each facility grade.
        """
        facility_grades = self.ccf_data["facility_grade"].unique()
        if len(facility_grades) > 20:
            raise ValidationError("More than 20 facility grades.")

        grade_level_reports = {}

        # Perform tests in each grade level
        for grade in facility_grades:
            # Get only that part of data
            data_slice = self.ccf_data.loc[
                self.ccf_data["facility_grade"] == grade
            ]

            # Obtain validator inputs from data
            N = len(
                data_slice.loc[
                    (data_slice["marked"] == "ead_covered") |
                    (data_slice["marked"] == "ccf_covered")
                ]
            )
            m_ead = len(
                data_slice.loc[
                    (data_slice["marked"] == "ead_covered")
                ]
            )
            m_outliers = len(
                data_slice.loc[
                    (data_slice["marked"] == "outlier")
                ]
            )
            estimated_ccfs = np.array(
                data_slice['estimated_CCF_cohort'].loc[
                    (~ (data_slice['estimated_CCF_cohort'].isnull())) &
                    (~ (data_slice['realised_CCF'].isnull())) &
                    (data_slice['marked'] == "ccf_covered")
                ],
                dtype=float
            )
            realised_ccfs = np.array(
                data_slice['realised_CCF'].loc[
                    (~ (data_slice['estimated_CCF_cohort'].isnull())) &
                    (~ (data_slice['realised_CCF'].isnull())) &
                    (data_slice['marked'] == "ccf_covered")
                ],
                dtype=float
            )

            grade_level_reports[grade] = self.validator.predictive_power_ccf(
                estimated_ccfs=estimated_ccfs,
                realised_ccfs=realised_ccfs,
                N=N,
                m_ead=m_ead,
                m_outliers=m_outliers,
                floor=floor,
                test_level=test_level
            )

        return {
            "test": "CCF predictive ability",
            "report": grade_level_reports
        }

    def validate_ead_predictive_power(self, test_level=0.05):
        """
        Function that checks the predictive power of EAD estimation.
        The function tests whether the estimation is greater than
        realised values or not, where the realised values are the drawn amount.
        """
        drawn_amounts = self.ccf_data["drawn_amounts"].loc[
            (self.ccf_data["marked"] == "ead_covered")
        ]
        estimated_eads = self.ccf_data["estimated_EAD"].loc[
            (self.ccf_data["marked"] == "ead_covered")
        ]

        report = self.validator.predictive_power_ead(
            drawn_amounts=drawn_amounts,
            estimated_eads=estimated_eads,
            test_level=test_level
        )
        report.name = "EAD predictive ability"

        return report

    # --------------------
    # DISCRIMINATORY POWER
    # --------------------

    def validate_discriminatory_power(self):
        return {
            "test": "Discriminatory power test",
            "report": "Validator not implemented"
        }

    # ---------------------
    # PORTFOLIO LEVEL TESTS
    # ---------------------

    def validate_assignment_process_portfolio(self):
        """
        Validates the assignent process at a portfolio level.

        Requires columns `estimated_CCF_cohort`, `estimated_CCF_fixed_horizon`,
        `estimated_EAD`.
        """
        M = len(self.ccf_data)
        m_miss = len(
            self.ccf_data.loc[
                (self.ccf_data["estimated_CCF_cohort"].isnull()) |
                (self.ccf_data["estimated_CCF_fixed_horizon"].isnull()) |
                (self.ccf_data["estimated_EAD"].isnull())
            ]
        )

        report = self.validator.assignment_process_portfolio(
            m_miss=m_miss,
            M=M
        )
        report.name = "Assignment process (portfolio)"

        return report

    def validate_ccf_portfolio_distribution(self):
        return {
            "test": "CCF portfolio distribution",
            "report": "Validator not implemented"
        }

    def validate_ead_application_portfolio(self):
        """
        Validates the application portfolio statistics for EAD
        covered facilities.

        Also reports the drawn amount for the portfolio, exposure at
        the beginning and end of the observation period, and total
        estimated EAD.
        """
        m_ead = len(
            self.ccf_data.loc[
                (self.ccf_data["marked"] == "ead_covered")
            ]
        )
        estimated_EADs = self.ccf_data.loc[
            (self.ccf_data["marked"] == "ead_covered")
        ]["estimated_EAD"]
        exposure_start = self.ccf_data["exposure_at_beginning"]
        exposure_end = self.ccf_data["exposure_at_end"]
        sum_drawings = self.ccf_data["drawn_amount"]

        report = self.validator.ead_application_portfolio(
            m_ead=m_ead,
            estimated_EADs=estimated_EADs,
            sum_drawings=sum_drawings,
            exposure_start=exposure_start,
            exposure_end=exposure_end
        )
        report.name = "EAD application portfolio"

        return report

    def run_validation_tests(self):
        results = []

        # Back testing tests
        results.append(self.validate_assignment_process_back_testing())
        results.append(self.validate_ead_covered_facilities())

        # # Predictive ability tests
        # results.append(self.validate_ccf_predictive_power())
        # results.append(self.validate_ead_predictive_power())

        # # Discriminatory power test
        results.append(self.validate_discriminatory_power())

        # # Application portfolio tests
        results.append(self.validate_assignment_process_portfolio())
        results.append(self.validate_ccf_portfolio_distribution())
        results.append(self.validate_ead_application_portfolio())

        return results
