"""Exception hierarchy for the project.

Every failure the program can anticipate is raised as a subclass of
:class:`CallMeMaybeError`. The CLI catches that single base class, prints
the message to ``stderr`` and exits with a clean, non-zero status code
instead of dumping a traceback.
"""


class CallMeMaybeError(Exception):
    """Base class for every expected, user-facing error."""


class ConfigError(CallMeMaybeError):
    """Invalid command-line options or configuration."""


class InputFileError(CallMeMaybeError):
    """An input file is missing, unreadable, malformed or invalid."""


class OutputFileError(CallMeMaybeError):
    """The output file could not be serialised or written."""


class TokenizerError(CallMeMaybeError):
    """The vocabulary or the tokenizer backend is unusable."""
