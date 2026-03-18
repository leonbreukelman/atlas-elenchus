"""
Retry wrapper for litellm.completion().
Exponential backoff on timeout and API errors.
"""

import asyncio
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

    Retries on: Timeout, APIError, APIConnectionError, RateLimitError.
    Backoff schedule (defaults): 5s, 15s, 45s.
    On final failure, re-raises the original exception.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return litellm.completion(**kwargs)
        except (litellm.Timeout, litellm.APIError, litellm.APIConnectionError, litellm.RateLimitError) as e:
            last_exception = e
            if attempt < max_retries:
                wait = backoff_base * (backoff_multiplier ** attempt)
                print(f"  [llm] Retry {attempt + 1}/{max_retries} after {type(e).__name__}: waiting {wait:.0f}s")
                time.sleep(wait)
            else:
                raise


async def acompletion_with_retry(
    *,
    max_retries: int = 3,
    backoff_base: float = 5.0,
    backoff_multiplier: float = 3.0,
    **kwargs,
):
    """
    Async version of completion_with_retry.

    Retries on: Timeout, APIError, APIConnectionError, RateLimitError.
    Backoff schedule (defaults): 5s, 15s, 45s.
    On final failure, re-raises the original exception.
    """
    for attempt in range(max_retries + 1):
        try:
            return await litellm.acompletion(**kwargs)
        except (litellm.Timeout, litellm.APIError, litellm.APIConnectionError, litellm.RateLimitError) as e:
            if attempt < max_retries:
                await asyncio.sleep(backoff_base * (backoff_multiplier ** attempt))
            else:
                raise
