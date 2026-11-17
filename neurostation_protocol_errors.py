"""Shared protocol validation error, independent of acquisition and Qt."""

class SSVEPProtocolError(ValueError):
    """A protocol cannot be executed safely."""
