"""
This module contains code to validate CCF models.
"""

import numpy as np
from scipy.stats import ttest_1samp

from crm_validator.exceptions import ValidationError
from crm_validator.report import PlotParams, SubReport


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
        if m_ex > m_b:
            raise ValidationError("m_ex is greater than m_b.")
        if N > m_b - m_ex:
            raise ValidationError("N is greater than m_b - m_ex.")

        # Create report
        return SubReport(
            passed=True,
            metrics={
                "m_ex/m_b": m_ex / m_b
            },
            reports={
                "m_ex": m_ex,
                "m_b": m_b,
                "N": N
            },
            plots=[
                PlotParams(
                    values=[m_ex, N],
                    labels=["Exclusions", "Number of facilities"],
                    kind="pie",
                    title="Exclusions in dataset"
                )
            ]
        )

    def ead_covered_facilities(
        self,
        m_ead: int,
        N: int
    ) -> dict:
        """
        This function reports validates the facilities covered by an EAD
        approach. Summary statistic reported is `m_ead / N`.
        """
        if m_ead > N:
            raise ValidationError("m_ead is greater than N.")

        return SubReport(
            passed=True,
            metrics={
                "m_ead/N": m_ead / N
            },
            reports={
                "m_ead": m_ead,
                "N": N
            },
            plots=[
                PlotParams(
                    values=[m_ead, N - m_ead],
                    labels=["EAD Covered", "CCF Covered"],
                    kind="pie",
                    title="EAD covered facilities"
                )
            ]
        )

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

        return SubReport(
            passed=passed,
            metrics={
                "p-value": p_value,
                "T-statistic": T_statistic,
                "N - m_ead": N - m_ead
            },
            reports={
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
            },
            plots=[
                PlotParams(
                    values=[estimated_ccfs, realised_ccfs],
                    labels=["Estimated CCFs", "Realised CCFs"],
                    kind="double_hist",
                    title="Estimated vs Realised CCF values"
                )
            ]
        )

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

        return SubReport(
            passed=passed,
            metrics={
                "p-value": p_value,
                "T-statistic": T_statistic
            },
            reports={
                "M_ead": m_ead,
                "Sum of estimated EAD": estimated_eads.sum(),
                "Sum of drawn amounts": drawn_amounts.sum(),
                "T statistic": T_statistic,
                "p-value": p_value,
                "Variance": (estimated_eads - drawn_amounts).var()
            }
        )

    # --------------------
    # DISCRIMINATORY POWER
    # --------------------

    def discriminatory_power(self):
        # TODO
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
        if m_miss > M:
            raise ValidationError("M_miss is greater than M.")

        return SubReport(
            passed=True,
            metrics={
                "m_miss/M": m_miss / M
            },
            reports={
                "m_miss": m_miss,
                "M": M,
                "m_miss / M": m_miss / M
            },
            plots=[
                PlotParams(
                    values=[m_miss, M - m_miss],
                    labels=["Missing", "Non-missing"],
                    kind="pie"
                )
            ]
        )

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
        return SubReport(
            passed=True,
            metrics={
                "Total facilities": m_ead,
                "Total estimated EAD": sum(estimated_EADs),
                "Sum of drawings": sum(sum_drawings),
                "Total exposure (start)": sum(exposure_start),
                "Total exposure (end)": sum(exposure_end)
            },
            reports={
                "Total facilities": m_ead,
                "Total estimated EAD": sum(estimated_EADs),
                "Sum of drawings": sum(sum_drawings),
                "Total exposure (start)": sum(exposure_start),
                "Total exposure (end)": sum(exposure_end)
            }
        )
