# src/exceptions/body_exceptions.py


class BodyValidationError(Exception):
    """Exception raised when there is a validation error in the request body."""

    def __init__(self, message: str = "Invalid request body"):
        super().__init__(message)
        self.message = message


class InvalidTextError(BodyValidationError):
    """Exception raised when the input text is invalid."""

    def __init__(self, message: str = "Invalid text input"):
        super().__init__(message)
        self.message = message


class TextProcessingError(BodyValidationError):
    """Exception raised when there is an error processing the text."""

    def __init__(self, message: str = "Error processing text"):
        super().__init__(message)
        self.message = message


class QueryValidationError(BodyValidationError):
    """Exception raised when there is a validation error in the query request body."""

    def __init__(self, message: str = "Invalid query request body"):
        super().__init__(message)
        self.message = message
