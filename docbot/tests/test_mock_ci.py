import unittest
from unittest.mock import patch, MagicMock
import tempfile
import json
import os
from pathlib import Path
import shutil

from docbot.ci import run_ci_index


class TestCIIndex(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_dir = Path(self.temp_dir) / ".chroma"
        self.repo_dir = Path(self.temp_dir) / "repo"
        self.repo_dir.mkdir()

        # Write dummy event payload
        self.event_file = Path(self.temp_dir) / "event.json"
        self.event_data = {
            "repository": {"full_name": "test_owner/test_repo"},
            "pull_request": {
                "number": 42,
                "title": "Fix some docs",
                "body": "This PR updates the programmers guide.",
            },
        }
        with open(self.event_file, "w") as f:
            json.dump(self.event_data, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch("docbot.ci.requests.get")
    def test_run_ci_index(self, mock_get):
        # Mock GitHub API response for PR comments
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"user": {"login": "developer1"}, "body": "Please review this doc edit."}
        ]
        mock_get.return_value = mock_response

        # Set environment variables
        os.environ["GITHUB_EVENT_PATH"] = str(self.event_file)
        os.environ["GITHUB_TOKEN"] = "mock_token"

        try:
            # Run the CI indexer
            run_ci_index(
                repo_dir=self.repo_dir,
                db_dir=self.db_dir,
                database_name="test_ci_db",
            )

            # Verify database directory was created
            self.assertTrue(self.db_dir.exists())

            # Load the Chroma DB collection to verify content
            import chromadb
            from docbot.common import get_chroma_settings

            chroma_settings = get_chroma_settings(self.db_dir)
            client = chromadb.PersistentClient(
                path=str(self.db_dir), settings=chroma_settings
            )

            # Check collection is created
            collections = {c.name for c in client.list_collections()}
            self.assertIn("test_ci_db", collections)

            collection = client.get_collection("test_ci_db")
            # Verify there is 1 document/node in the collection
            self.assertGreater(collection.count(), 0)

        finally:
            # Clean up env
            os.environ.pop("GITHUB_EVENT_PATH", None)
            os.environ.pop("GITHUB_TOKEN", None)


if __name__ == "__main__":
    unittest.main()
