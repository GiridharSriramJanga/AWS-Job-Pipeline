"""
test_integration.py
────────────────────
Integration tests — multiple components working together.
S3 is mocked with moto. SMTP is mocked with unittest.mock.

Test count:
    load_and_filter_from_s3  4
    send_alert               5
    Full cleaning pipeline   2
    ──────────────────────────
    Total                   11
"""

import os
import pytest
import pandas as pd
import numpy as np
from io import BytesIO
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timedelta

# ── Set test env vars before any pipeline imports ─────────────────────────────
os.environ.setdefault('GROQ_API_KEY',        'test_groq_key')
os.environ.setdefault('ALERT_SENDER_MAIL',   'test@test.com')
os.environ.setdefault('ALERT_RECEIVER_MAIL', 'test@test.com')
os.environ.setdefault('GMAIL_APP_PASSWORD',  'test_password')

from pipeline.resume_matcher import load_and_filter_from_s3, DATE_RANGES
from pipeline.notifications  import send_alert
from pipeline.Jobsdata_cleaning_final_version import (
    job_title_clean,
    clean_salary_data,
    extract_experience_years,
    seniority,
    education_extract,
)


# ══════════════════════════════════════════════════════════════════════════════
# load_and_filter_from_s3 — 4 tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadAndFilterFromS3:
    """
    Uses a mocked boto3 S3 client that returns a real parquet file
    in memory. Verifies that metadata filters work correctly without
    touching any real AWS infrastructure.
    """

    def _make_parquet_bytes(self, df: pd.DataFrame) -> bytes:
        """Serialise a dataframe to parquet bytes for S3 mock."""
        buf = BytesIO()
        df.to_parquet(buf)
        buf.seek(0)
        return buf.read()

    def _build_mock_s3(self, df: pd.DataFrame) -> MagicMock:
        """Build a mock S3 client that serves the given dataframe as parquet."""
        parquet_bytes = self._make_parquet_bytes(df)

        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'Embeddings_data/jobs_with_embeddings_11-11-2025.parquet'}
            ]
        }

        def fake_download_fileobj(bucket, key, fileobj):
            fileobj.write(parquet_bytes)
            fileobj.seek(0)

        mock_client.download_fileobj.side_effect = fake_download_fileobj
        return mock_client

    def test_role_filter_returns_only_matching_role(self, sample_cleaned_df):
        """
        When role='Data Analyst' is passed, only Data Analyst rows
        are returned regardless of what else is in the file.
        """
        sample_cleaned_df['chunk_embeddings'] = [
            [[0.1] * 384] for _ in range(len(sample_cleaned_df))
        ]
        # Add a recent date so date filter passes
        sample_cleaned_df['Scrapped_date'] = datetime.now().strftime('%m-%d-%Y')

        mock_s3 = self._build_mock_s3(sample_cleaned_df)
        result  = load_and_filter_from_s3(mock_s3, 'Data Analyst', 'Last month', None)

        assert not result.empty
        assert all(result['Job_Title_Cleaned'] == 'Data Analyst')

    def test_date_filter_excludes_old_records(self, sample_cleaned_df):
        """
        Records with Scrapped_date older than the selected range
        should be excluded from results.
        """
        sample_cleaned_df['chunk_embeddings'] = [
            [[0.1] * 384] for _ in range(len(sample_cleaned_df))
        ]
        # Set all dates to 6 months ago — outside any filter window
        old_date = (datetime.now() - timedelta(days=180)).strftime('%m-%d-%Y')
        sample_cleaned_df['Scrapped_date'] = old_date

        mock_s3 = self._build_mock_s3(sample_cleaned_df)
        result  = load_and_filter_from_s3(mock_s3, 'Data Analyst', 'Last week', None)

        assert result.empty

    def test_empty_s3_returns_empty_dataframe(self, mock_s3_empty):
        """When S3 has no files, an empty dataframe is returned."""
        result = load_and_filter_from_s3(mock_s3_empty, 'Data Analyst', 'Last week', None)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_cross_file_duplicates_removed(self, sample_cleaned_df):
        """
        Same job appearing in multiple S3 files is deduplicated.
        Company + Job Title + Location combination must be unique.
        """
        sample_cleaned_df['chunk_embeddings'] = [
            [[0.1] * 384] for _ in range(len(sample_cleaned_df))
        ]
        sample_cleaned_df['Scrapped_date'] = datetime.now().strftime('%m-%d-%Y')

        parquet_bytes = self._make_parquet_bytes(sample_cleaned_df)

        # Simulate two files with identical content — same jobs scraped twice
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'Embeddings_data/jobs_11-10-2025.parquet'},
                {'Key': 'Embeddings_data/jobs_11-11-2025.parquet'},
            ]
        }

        def fake_download_fileobj(bucket, key, fileobj):
            fileobj.write(parquet_bytes)
            fileobj.seek(0)

        mock_client.download_fileobj.side_effect = fake_download_fileobj

        result = load_and_filter_from_s3(
            mock_client, 'Data Analyst', 'Last month', None
        )

        # Result should have same number of rows as one file — no duplicates
        expected_data_analyst_rows = (
            sample_cleaned_df['Job_Title_Cleaned'] == 'Data Analyst'
        ).sum()

        assert len(result) == expected_data_analyst_rows


# ══════════════════════════════════════════════════════════════════════════════
# send_alert — 5 tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSendAlert:
    """
    Tests that send_alert constructs the correct email for each stage.
    SMTP is mocked — no real emails are sent during testing.
    """

    def _get_sent_message_body(self, mock_smtp_instance) -> str:
        """Extract the body of the email that was sent via mock SMTP."""
        send_call = mock_smtp_instance.send_message.call_args
        msg       = send_call[0][0]
        return msg.get_content()

    def _get_sent_subject(self, mock_smtp_instance) -> str:
        send_call = mock_smtp_instance.send_message.call_args
        msg       = send_call[0][0]
        return msg['Subject']

    @patch('smtplib.SMTP')
    def test_raw_failed_alert_has_correct_subject(self, mock_smtp):
        """raw_failed stage sends email with PIPELINE ABORTED in subject."""
        mock_smtp.return_value.__enter__ = lambda s: s
        mock_smtp.return_value.__exit__  = MagicMock(return_value=False)

        send_alert(
            stage='raw_failed',
            details={'failed_checks': {'null_threshold': 'Salary 30% null'}}
        )

        subject = self._get_sent_subject(mock_smtp.return_value)
        assert 'ABORTED' in subject
        assert 'Raw' in subject

    @patch('smtplib.SMTP')
    def test_clean_failed_alert_has_correct_subject(self, mock_smtp):
        """clean_failed stage sends email with PIPELINE WARNING in subject."""
        mock_smtp.return_value.__enter__ = lambda s: s
        mock_smtp.return_value.__exit__  = MagicMock(return_value=False)

        send_alert(
            stage='clean_failed',
            details={'failed_checks': {'Min_Salary_nulls': '25% null'}}
        )

        subject = self._get_sent_subject(mock_smtp.return_value)
        assert 'WARNING' in subject

    @patch('smtplib.SMTP')
    def test_pipeline_success_alert_has_correct_subject(self, mock_smtp):
        """pipeline_success stage sends email with SUCCESS in subject."""
        mock_smtp.return_value.__enter__ = lambda s: s
        mock_smtp.return_value.__exit__  = MagicMock(return_value=False)

        send_alert(
            stage='pipeline_success',
            details={'row_count': 42, 'roles_present': 'Data Analyst'}
        )

        subject = self._get_sent_subject(mock_smtp.return_value)
        assert 'SUCCESS' in subject

    @patch('smtplib.SMTP')
    def test_pipeline_crashed_alert_has_correct_subject(self, mock_smtp):
        """pipeline_crashed stage sends email with CRASHED in subject."""
        mock_smtp.return_value.__enter__ = lambda s: s
        mock_smtp.return_value.__exit__  = MagicMock(return_value=False)

        send_alert(
            stage='pipeline_crashed',
            details={
                'script'   : 'data_diagnostics.py',
                'error'    : 'FileNotFoundError',
                'traceback': 'Traceback...'
            }
        )

        subject = self._get_sent_subject(mock_smtp.return_value)
        assert 'CRASHED' in subject

    def test_missing_env_vars_raises_environment_error(self):
        """
        If required env vars are missing, EnvironmentError is raised
        before any SMTP connection is attempted.
        """
        with patch.dict(os.environ, {
            'ALERT_SENDER_MAIL'  : '',
            'ALERT_RECEIVER_MAIL': '',
            'GMAIL_APP_PASSWORD' : '',
        }):
            # Temporarily remove the keys
            env_backup = {}
            for key in ['ALERT_SENDER_MAIL', 'ALERT_RECEIVER_MAIL', 'GMAIL_APP_PASSWORD']:
                env_backup[key] = os.environ.pop(key, None)

            with pytest.raises(EnvironmentError):
                send_alert('raw_failed', {'failed_checks': {}})

            # Restore
            for key, val in env_backup.items():
                if val is not None:
                    os.environ[key] = val


# ══════════════════════════════════════════════════════════════════════════════
# Full cleaning pipeline — 2 tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFullCleaningPipeline:
    """
    Tests that running all cleaning functions in sequence on a sample
    dataframe produces a correctly shaped and typed output — the same
    sequence as in Jobsdata_cleaning_final_version.py's main block.
    """

    def _run_full_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replicate the cleaning pipeline logic on a given dataframe."""
        df = df.copy()

        # Deduplication
        df = df[~df.drop('Job_Role', axis=1).duplicated()]

        # Title cleaning
        df['Job_Title_Cleaned'] = df['Job Title'].apply(job_title_clean)
        df = df[~df['Job_Title_Cleaned'].isin(['Other', 'Intern'])]

        # Salary
        df = clean_salary_data(df)

        # State
        df['State'] = df['Location'].apply(
            lambda x: str(x).split(',')[1].strip() if ',' in str(x) else None
        )

        # Experience
        with patch(
            'pipeline.Jobsdata_cleaning_final_version.groq_client'
        ) as mock_groq:
            mock_groq.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content='-1'))]
            )
            df['Experience'] = df['Job Description'].apply(extract_experience_years)

        # Seniority
        df['Seniority_simplified'] = df.apply(
            lambda row: seniority(row['Job Title'], row['Experience']), axis=1
        )

        # Education
        df['Education'] = df['Job Description'].apply(education_extract)

        return df

    def test_cleaning_pipeline_produces_expected_columns(self, sample_raw_df):
        """All cleaning steps produce the expected set of columns."""
        result = self._run_full_cleaning(sample_raw_df)

        expected_cols = [
            'Job_Title_Cleaned', 'Min_Salary', 'Max_Salary',
            'Avg Salary', 'State', 'Experience',
            'Seniority_simplified', 'Education'
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing expected column: {col}"

    def test_cleaning_pipeline_removes_duplicates(self, sample_raw_df):
        """Duplicate rows are removed during deduplication step."""
        # Inject exact duplicates
        duplicated_df = pd.concat(
            [sample_raw_df, sample_raw_df.iloc[:5]], ignore_index=True
        )
        result = self._run_full_cleaning(duplicated_df)
        assert len(result) == len(
            self._run_full_cleaning(sample_raw_df)
        )
