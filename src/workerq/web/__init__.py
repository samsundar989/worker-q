"""The local web UI.

Nothing here is a new control channel. The webapp is a client, exactly like
the CLI: it reads the databases directly and writes only through
`GPUQService`, so the dispatcher still listens to nothing but the queue
database and the spec's rule against an unauthenticated network server holds.
"""

from workerq.web.server import DEFAULT_PORT, serve

__all__ = ["serve", "DEFAULT_PORT"]
