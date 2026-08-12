class AutoFineTuneError(Exception):
    """Base error."""


class FatalError(AutoFineTuneError):
    """Unrecoverable; CLI should exit non-zero immediately."""


class RoundError(AutoFineTuneError):
    """Recoverable at round level; orchestrator may replan."""
