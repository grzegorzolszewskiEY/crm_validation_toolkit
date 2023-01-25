"""
This module contains custom-defined errors.
"""


class ValidationError(Exception):
    """
    Raised when there's an issue with validation.
    """
    def __init__(
        self,
        message: str = "Could not validate."
    ) -> None:
        self.message = message
        super().__init__(self.message)
