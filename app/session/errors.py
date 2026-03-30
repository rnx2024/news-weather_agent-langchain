SESSION_UNAVAILABLE_MESSAGE = "Session can't be loaded or Session can't be retrieved."


class SessionStoreUnavailable(RuntimeError):
    """Raised when session storage is unavailable."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or SESSION_UNAVAILABLE_MESSAGE)
