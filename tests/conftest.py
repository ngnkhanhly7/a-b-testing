import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Point the API's SQLite storage at a throwaway file for tests, so test runs
# never touch (or get polluted by) the real data/platform.db, and each test
# session starts from a clean, empty database.
_TEST_DB_PATH = os.path.join(tempfile.gettempdir(), "ab_platform_test.db")
if os.path.exists(_TEST_DB_PATH):
    os.remove(_TEST_DB_PATH)
os.environ["AB_PLATFORM_DB_PATH"] = _TEST_DB_PATH

from src import storage  # noqa: E402

storage.init_db()
