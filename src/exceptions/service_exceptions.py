# src/exceptions/service_exceptions.py


class ServiceException(Exception):
    """Base class for service exceptions."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


# Document Loader Exceptions
class DocumentLoaderException(ServiceException):
    """Base class for document loader exceptions."""

    def __init__(
        self, message: str = "An error occurred in the document loader service."
    ):
        super().__init__(message)


class DocumentNotFoundException(DocumentLoaderException):
    """Raised when a document is not found."""

    def __init__(self, message: str = "Document not found."):
        super().__init__(message)


class DocumentReadException(DocumentLoaderException):
    """Raised when a document cannot be read."""

    def __init__(self, message: str = "Document could not be read."):
        super().__init__(message)


class DocumentInvalidParametersException(DocumentLoaderException):
    """Raised when invalid parameters are provided to the document loader."""

    def __init__(
        self, message: str = "Invalid parameters provided to the document loader."
    ):
        super().__init__(message)


# file service exceptions
class FileServiceException(ServiceException):
    """Base class for file service exceptions."""

    def __init__(self, message: str = "An error occurred in the file service."):
        super().__init__(message)


class FileAlreadyExistsException(FileServiceException):
    """Raised when a file already exists."""

    def __init__(self, message: str = "File already exists."):
        super().__init__(message)


class FileNotFoundException(FileServiceException):
    """Raised when a file is not found."""

    def __init__(self, message: str = "File not found."):
        super().__init__(message)


class FileReadException(FileServiceException):
    """Raised when a file cannot be read."""

    def __init__(self, message: str = "File could not be read."):
        super().__init__(message)


class FileWriteException(FileServiceException):
    """Raised when a file cannot be written."""

    def __init__(self, message: str = "File could not be written."):
        super().__init__(message)
