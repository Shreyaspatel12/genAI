"""
tools/http_client.py — Shared requests session with retry + exponential backoff.
Import get_session() anywhere instead of using requests directly.
"""
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

_RETRY_STRATEGY = Retry(
    total=3,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)

_HEADERS = {
    "User-Agent": "ChemAgentPipeline/0.1 (research; contact: your@email.com)",
    "Accept":     "application/json",
}


def get_session() -> requests.Session:
    """Return a configured requests Session."""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_RETRY_STRATEGY)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    session.headers.update(_HEADERS)
    return session


def safe_get(url: str, params: dict | None = None, session: requests.Session | None = None) -> dict:
    """
    GET url, return parsed JSON.
    Raises RuntimeError with a descriptive message on failure.
    """
    s = session or get_session()
    try:
        resp = s.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Timeout after {REQUEST_TIMEOUT}s: {url}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP {resp.status_code} from {url}: {e}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Request failed for {url}: {e}")
    except ValueError:
        raise RuntimeError(f"Non-JSON response from {url}: {resp.text[:200]}")
