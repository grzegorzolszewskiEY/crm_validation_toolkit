"""
This module contains code to validate CCF models.
"""

from crm_validator.constants import PASSED, METRIC, REPORT
from crm_validator.exceptions import ValidationError


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
        inputs = [m_ex, m_b, N]

        assert all(inputs), f"At least one input of {inputs} not provided."
        if m_ex > m_b:
            raise ValidationError("m_ex is greater than m_b.")
        if N > m_b - m_ex:
            raise ValidationError("N is greater than m_b - m_ex.")

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
        assert all(inputs), f"At least one input of {inputs} not provided."
        if m_ead > N:
            raise ValidationError("m_ead is greater than N.")

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
