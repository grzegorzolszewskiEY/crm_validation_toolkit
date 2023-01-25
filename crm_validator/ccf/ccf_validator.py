"""
This module contains code to validate CCF models.
"""

from constants import PASSED, METRIC, REPORT


class CCFValidator:
    """
    Main CCF Validator class.
    """
    def __init__(self):
        pass

    def assignment_process(
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
        try:
            assert m_ex is not None
            assert m_b is not None
            assert N is not None
            assert m_ex < m_b
            assert N < m_b - m_ex

            return {
                PASSED: True,
                METRIC: {
                    "name": "m_ex/m_b",
                    "value": m_ex / m_b
                },
                REPORT: {
                    "m_ex": m_ex,
                    "m_b": m_b,
                    "N": N
                }
            }

        except AssertionError as e:
            return {
                PASSED: False,
                METRIC: {
                    "name": "m_ex/m_b",
                    "value": m_ex / m_b
                },
                REPORT: {
                    "Error": e
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
        try:
            assert m_ead is not None
            assert N is not None
            assert m_ead < N

            return {
                PASSED: True,
                METRIC: {
                    "name": "m_ead/N",
                    "value": m_ead / N
                },
                REPORT: {
                    "m_ead": m_ead,
                    "N": N
                }
            }

        except AssertionError as e:
            return {
                PASSED: False,
                METRIC: {
                    "name": "m_ead/N",
                    "value": m_ead / N
                },
                REPORT: {
                    "Error": e
                }
            }
