"""Small cross-platform local protection helpers for user registry data."""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes


class SecureStorageError(RuntimeError):
    pass


def _dpapi_transform(data: bytes, *, decrypt: bool) -> bytes:
    class DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    source_buffer = ctypes.create_string_buffer(data)
    source = DATA_BLOB(len(data), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_byte)))
    target = DATA_BLOB()
    if decrypt:
        ok = crypt32.CryptUnprotectData(ctypes.byref(source), None, None, None, None, 0, ctypes.byref(target))
    else:
        ok = crypt32.CryptProtectData(ctypes.byref(source), "NeuroStation user registry", None, None, None, 0, ctypes.byref(target))
    if not ok:
        raise SecureStorageError(f"DPAPI failed: {ctypes.get_last_error()}")
    try:
        return ctypes.string_at(target.pbData, target.cbData)
    finally:
        kernel32.LocalFree(target.pbData)


def protect(data: bytes) -> tuple[bytes, str]:
    if os.name == "nt":
        return b"NS-DPAPI-1\n" + _dpapi_transform(data, decrypt=False), "dpapi"
    return b"NS-PLAINTEXT-1\n" + data, "plaintext-fallback"


def unprotect(data: bytes) -> tuple[bytes, str]:
    if data.startswith(b"NS-DPAPI-1\n"):
        if os.name != "nt":
            raise SecureStorageError("DPAPI registry can only be opened on Windows")
        return _dpapi_transform(data.split(b"\n", 1)[1], decrypt=True), "dpapi"
    if data.startswith(b"NS-PLAINTEXT-1\n"):
        return data.split(b"\n", 1)[1], "plaintext-fallback"
    # Legacy sidecars were never encrypted; preserve readability.
    return data, "legacy-plaintext"
