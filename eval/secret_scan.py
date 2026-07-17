"""Re-export from sift/core/secret_scan.py.

The implementation now lives in production code so it runs in both the eval
harness and the real review pipeline. This shim keeps existing eval imports
working without change.
"""
from sift.core.secret_scan import scan_diff_for_secrets  # noqa: F401
