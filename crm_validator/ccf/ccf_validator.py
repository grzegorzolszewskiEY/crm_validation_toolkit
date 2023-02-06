"""
This module contains code to validate CCF models.
"""

import numpy as np
from crm_validator.constants import METRIC, PASSED, REPORT
from crm_validator.exceptions import ValidationError
from scipy.stats import ttest_1samp


class CCFValidator:
    """
    Main CCF Validator class.
    """
    def __init__(self):
        pass

    # --------------------
    # BACK TESTING REPORTS
    # --------------------

    def assignment_process_back_testing(
        self,
        m_ex: int,
        m_b: int,
        N: int
    ) -> dict:
        """
        This function asserts the provision of number of facilities before
        exclusions (`m_b`), number of facilities excluded (`m_ex`), and number
        of facilities for validation (`N`).
        """
        inputs = [m_ex, m_b, N]

        assert all(inputs), "At least one input not provided."
        if m_ex > m_b:
            raise ValidationError("m_ex is greater than m_b.")
        if N > m_b - m_ex:
            raise ValidationError("N is greater than m_b - m_ex.")

        return {
            PASSED: True,
            METRIC: {
                "m_ex/m_b": m_ex / m_b
            },
            REPORT: {
                "m_ex": m_ex,
                "m_b": m_b,
                "N": N
            }
        }

    def ead_covered_facilities(
        self,
        m_ead: int,
        N: int
    ) -> dict:
        """
        This function reports validates the facilities covered by an EAD
        approach. Summary statistic reported is `m_ead / N`.
        """
        inputs = [m_ead, N]

        # Validations
        assert all(inputs), "At least one input not provided."
        if m_ead > N:
            raise ValidationError("m_ead is greater than N.")

        return {
            PASSED: True,
            METRIC: {
                "m_ead/N": m_ead / N
            },
            REPORT: {
                "m_ead": m_ead,
                "N": N
            }
        }

    # ------------------------
    # PREDICTIVE ABILITY TESTS
    # ------------------------

    def predictive_power_ccf(
        self,
        estimated_ccfs: np.ndarray,
        realised_ccfs: np.ndarray,
        N: int,
        m_ead: int,
        m_outliers: int,
        floor: float = None,
        test_level: float = 0.05
    ):
        """
        Validation of estimated CCFs. The aim is to ensure that estimates
        of CCF are higher than the real values.
        """
        # TODO: Ensure inputs are given

        # Check whether number of EAD covered facilities is greater than
        # facilities available
        if m_ead > N:
            raise ValidationError("m_ead is greater than N.")

        # Perform t-test
        ttest_result = ttest_1samp(
            estimated_ccfs,
            realised_ccfs.mean(),
            alternative="less"
        )

        T_statistic = ttest_result.statistic
        p_value = ttest_result.pvalue

        # Calculate percentage of floored CCF values
        floor_perc = 0
        if floor:
            floored_count = (estimated_ccfs == floor).sum()
            floor_perc = floored_count / len(estimated_ccfs)

        passed = True
        if p_value > test_level:
            passed = False

        return {
            PASSED: passed,
            METRIC: {
                "p-value": p_value,
                "T-statistic": T_statistic,
                "N - m_ead": N - m_ead
            },
            REPORT: {
                "N - m_ead": N - m_ead,
                "Estimated CCF average": estimated_ccfs.mean(),
                "Realised CCF average": realised_ccfs.mean(),
                "Floor percentage": floor_perc,
                "Floor used": floor,
                "Distribution of CCF": {
                    "min": estimated_ccfs.min(),
                    "05%": np.percentile(estimated_ccfs, 5),
                    "10%": np.percentile(estimated_ccfs, 10),
                    "25%": np.percentile(estimated_ccfs, 25),
                    "50%": np.percentile(estimated_ccfs, 50),
                    "75%": np.percentile(estimated_ccfs, 75),
                    "90%": np.percentile(estimated_ccfs, 90),
                    "95%": np.percentile(estimated_ccfs, 95),
                    "max": estimated_ccfs.max()
                },
                "T statistic": T_statistic,
                "p-value": p_value,
                "Variance": (estimated_ccfs - realised_ccfs).var(),
                "Outliers": m_outliers
            }
        }

    def predictive_power_ead(
        self,
        drawn_amounts: np.ndarray,
        estimated_eads: np.ndarray,
        test_level: float = 0.05
    ) -> dict:
        """
        Function to validate EAD assignment. The aim is to ensure that
        estimated EADs are statistically greater than real drawn amounts.
        """
        inputs = [drawn_amounts, estimated_eads]
        assert all(inputs), "At least one input not provided."

        m_ead = len(estimated_eads)
        assert len(drawn_amounts) == m_ead, "Arrays not equal length."

        ttest_result = ttest_1samp(
            estimated_eads,
            drawn_amounts.mean(),
            alternative="less"
        )

        T_statistic = ttest_result.statistic
        p_value = ttest_result.pvalue

        passed = True
        if p_value < test_level:
            passed = False

        return {
            PASSED: passed,
            METRIC: {
                "p-value": p_value,
                "T-statistic": T_statistic
            },
            REPORT: {
                "M_ead": m_ead,
                "Sum of estimated EAD": estimated_eads.sum(),
                "Sum of drawn amounts": drawn_amounts.sum(),
                "T statistic": T_statistic,
                "p-value": p_value,
                "Variance": (estimated_eads - drawn_amounts).var()
            }
        }

    # --------------------
    # DISCRIMINATORY POWER
    # --------------------

    def discriminatory_power(self):
        return

    # ---------------------
    # PORTFOLIO LEVEL TESTS
    # ---------------------

    def assignment_process_portfolio(
        self,
        m_miss: int,
        M: int
    ) -> dict:
        """
        Portfolio-level assignment process validation.
        Checks number of missing values, and total values
        available for validation.
        """
        inputs = [m_miss, M]

        assert all(inputs), "At least one input not provided."
        if m_miss > M:
            raise ValidationError("M_miss is greater than M.")

        return {
            PASSED: True,
            METRIC: {
                "m_miss/M": m_miss / M
            },
            REPORT: {
                "m_miss": m_miss,
                "M": M,
                "m_miss / M": m_miss / M
            }
        }

    def ccf_portfolio_distribution(
        self,
        N: int,
        avg_estimated_ccf: float,
        avg_line_usage: float,
        avg_undrawn_amt: float,
    ):
        return

    def ead_application_portfolio(
        self,
        m_ead: int,
        estimated_EADs: float,
        sum_drawings: float,
        exposure_start: float,
        exposure_end: float,
    ):
        inputs = [
            m_ead,
            estimated_EADs,
            sum_drawings,
            exposure_start,
            exposure_end
        ]

        assert all(inputs), "At least one input not provided."

        return {
            PASSED: True,
            METRIC: {
                "Total facilities": m_ead,
                "Total estimated EAD": sum(estimated_EADs),
                "Sum of drawings": sum(sum_drawings),
                "Total exposure (start)": sum(exposure_start),
                "Total exposure (end)": sum(exposure_end)
            },
            REPORT: {
                "Total facilities": m_ead,
                "Total estimated EAD": sum(estimated_EADs),
                "Sum of drawings": sum(sum_drawings),
                "Total exposure (start)": sum(exposure_start),
                "Total exposure (end)": sum(exposure_end)
            }
        }
