"""
Module was created as a wrapper to preprocess data and handle
inputs to the validator, so multiple calculations of inputs to
validator are not required.
"""

import pandas as pd
from crm_validator.ccf.ccf_validator import CCFValidator


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

    def validate_assignment_process(self):
        """
        Validates the assignment process.

        Total number of facilities is length of dataset.
        Number of exclusions are marked in dataset as "outlier" or
            "process deficiency".
        Number of final facilities is marked as either "ead covered"
            or "ccf covered"
        """
        m_b = len(self.ccf_data)
        m_ex = self.ccf_data.iloc[
            self.ccf_data.marked == "outlier" or
            self.ccf_data.marked == "process deficiency"
        ]
        N = self.ccf_data.iloc[
            self.ccf_data.marked == "ead covered" or
            self.ccf_data.marked == "ccf covered"
        ]

        return self.validator.validate_assignment_process(
            m_b=m_b,
            m_ex=m_ex,
            N=N
        )
