"""
Retry wrapper for litellm.completion().
Exponential backoff on timeout and API errors.
"""

import time
import litellm


def completion_with_retry(
    *,
    max_retries: int = 3,
    backoff_base: float = 5.0,
    backoff_multiplier: float = 3.0,
    **kwargs,
):
    """
    Call litellm.completion() with exponential backoff on transient failures.

    Retries on: Timeout, APIError, APIConnectionError.
    Backoff schedule (defaults): 5s, 15s, 45s.
    On final failure, re-raises the original exception.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return litellm.completion(**kwargs)
        except (litellm.Timeout, litellm.APIError, litellm.APIConnectionError) as e:
            last_exception = e
            if attempt < max_retries:
                wait = backoff_base * (backoff_multiplier ** attempt)
                print(f"  [llm] Retry {attempt + 1}/{max_retries} after {type(e).__name__}: waiting {wait:.0f}s")
                time.sleep(wait)
            else:
                raise
