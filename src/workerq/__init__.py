"""worker-q - resource broker for machines shared by AI coding agents."""

__version__ = "1.2.0"

# Pinned backend identity. Isolated here so swapping/upgrading the execution
# backend is a one-line change (spec section 2).
BACKEND_NAME = "local_dispatcher"
BACKEND_VERSION = "1.0.0"

# Upstream GPU Task Spooler tag this project would pin on Linux. Retained so the
# future TaskSpoolerBackend has a single source of truth for its version.
TASK_SPOOLER_PINNED_TAG = "v2.0.0"

__all__ = ["__version__", "BACKEND_NAME", "BACKEND_VERSION", "TASK_SPOOLER_PINNED_TAG"]
