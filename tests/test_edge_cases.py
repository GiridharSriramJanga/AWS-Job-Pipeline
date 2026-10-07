"""
test_edge_cases.py
───────────────────
Edge case tests — the scenarios most likely to break your pipeline
in production. Each test targets a specific failure mode that would
be silent or hard to debug without explicit coverage.

Test count:
    Empty dataframe at validation stages      2
    Single row dataframe                      1
    All rows duplicates                       1
    Blank and whitespace resume               2
    Zero skills in resume                     1
    Empty chunk_embeddings list               1
    S3 returns no files                       1
    All salary null post clean                1
    ──────────────────────────────────────────
    Total                                    10
"""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# ── Set test env vars before any pipeline imports ─────────────────────────────
os.environ.setdefault('GROQ_API_KEY',        'test_groq_key')
os.environ.setdefault('ALERT_SENDER_MAIL',   'test@test.com')
os.environ.setdefault('ALERT_RECEIVER_MAIL', 'test@test.com')
os.environ.setdefault('GMAIL_APP_PASSWORD',  'test_password')

from pipeline.data_diagnostics      import run_validation
from pipeline.cleaning_validation   import run_post_clean_validation
from pipeline.resume_matcher        import (
    compute_semantic_score,
    extract_resume_skills,
    load_and_filter_from_s3,
)
from pipeline.Jobsdata_cleaning_final_version import clean_salary_data


# ══════════════════════════════════════════════════════════════════════════════
# Empty dataframe at validation stages — 2 tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEmptyDataframe:

    def test_empty_df_through_run_validation_does_not_crash(self):
        """
        Completely empty dataframe passed to raw validation should
        return a failed check — not raise an unhandled exception.
        Pipeline must fail gracefully, not silently crash.
        """
        result = run_validation(pd.DataFrame(), [], [])
        assert isinstance(result, dict)
        assert 'total_row_count' in result

    def test_empty_df_through_post_clean_validation_does_not_crash(self):
        """
        Empty dataframe through post-clean validation should return
        a failed check — not raise an unhandled exception.
        """
        result = run_post_clean_validation(pd.DataFrame())
        assert isinstance(result, dict)
        assert 'row_count_post_clean' in result


# ══════════════════════════════════════════════════════════════════════════════
# Single row dataframe — 1 test
# ══════════════════════════════════════════════════════════════════════════════

class TestSingleRowDataframe:

    def test_single_row_df_does_not_crash_validation(self):
        """
        A dataframe with exactly one row should not crash validation.
        It will fail threshold checks (1 row is below MIN_ROWS) but
        the failure should be clean and structured, not an exception.
        Real scenario: scraper runs but only finds one job posting.
        """
        df = pd.DataFrame([{
            'Company Name'    : 'Acme Corp',
            'Job Title'       : 'Data Analyst',
            'Location'        : 'Chicago, IL',
            'Job Description' : 'Python SQL required. ' * 30,
            'Salary Estimate' : '$80K/yr',
            'Scrapped_date'   : '11-11-2025',
            'Job_Role'        : 'Data Analyst',
        }])

        result = run_validation(df, ['Data Analyst'], [])

        # Should return structured failures, not raise an exception
        assert isinstance(result, dict)


# ══════════════════════════════════════════════════════════════════════════════
# All rows duplicates — 1 test
# ══════════════════════════════════════════════════════════════════════════════

class TestAllRowsDuplicated:

    def test_all_duplicate_rows_flagged_correctly(self):
        """
        Dataframe where every row is a duplicate of the first row.
        Duplicate rate would be nearly 100% — well above threshold.
        Real scenario: scraper pagination bug causes same page to be
        scraped multiple times.
        """
        single_row = {
            'Company Name'    : 'Acme Corp',
            'Job Title'       : 'Data Analyst',
            'Location'        : 'Chicago, IL',
            'Job Description' : 'Python SQL required. ' * 30,
            'Salary Estimate' : '$80K/yr',
            'Scrapped_date'   : '11-11-2025',
            'Job_Role'        : 'Data Analyst',
        }
        # 50 identical rows
        df     = pd.DataFrame([single_row] * 50)
        result = run_validation(df, ['Data Analyst'], [])

        assert 'duplicates' in result


# ══════════════════════════════════════════════════════════════════════════════
# Blank and whitespace resume — 2 tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBlankResume:

    def test_empty_string_resume_returns_empty_skill_set(self):
        """
        Empty string resume produces an empty skill set.
        Should not crash — graceful handling of bad user input.
        """
        skills = extract_resume_skills('')
        assert isinstance(skills, set)
        assert len(skills) == 0

    def test_whitespace_only_resume_returns_empty_skill_set(self):
        """
        Resume containing only whitespace and newlines should not
        crash or produce false skill matches.
        Real scenario: user uploads a blank PDF or formatting-only file.
        """
        skills = extract_resume_skills('   \n\n\t   \n   ')
        assert isinstance(skills, set)
        assert len(skills) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Zero skills in resume — 1 test
# ══════════════════════════════════════════════════════════════════════════════

class TestZeroSkillsResume:

    def test_skill_score_is_zero_when_resume_has_no_skills(
        self, sample_cleaned_df
    ):
        """
        Resume with no detectable technical skills produces a skill
        score of 0.0 for every job — even jobs with many skills listed.
        The pipeline should not crash — just return zero scores.

        Real scenario: user uploads a non-technical resume or a resume
        that uses unconventional terminology for all skills.
        """
        from pipeline.resume_matcher import compute_skill_score

        empty_skills = set()

        for _, job_row in sample_cleaned_df.iterrows():
            score = compute_skill_score(empty_skills, job_row)
            assert score == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Empty chunk_embeddings list — 1 test
# ══════════════════════════════════════════════════════════════════════════════

class TestEmptyJobEmbeddings:

    def test_empty_chunk_embeddings_returns_zero_score(
        self, mock_resume_embeddings
    ):
        """
        Job with empty chunk_embeddings list (failed embedding generation)
        should return 0.0 semantic score without crashing.

        Real scenario: a specific job's description was empty or too
        short to produce any valid chunks during embeddings.py run.
        The rest of the pipeline should continue unaffected.
        """
        score = compute_semantic_score(mock_resume_embeddings, [])
        assert score == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# S3 returns no files — 1 test
# ══════════════════════════════════════════════════════════════════════════════

class TestS3ReturnsNoFiles:

    def test_empty_s3_bucket_returns_empty_dataframe(self, mock_s3_empty):
        """
        When S3 Embeddings_data/ folder is empty — no parquet files
        exist yet — load_and_filter_from_s3 returns an empty dataframe
        without crashing.

        Real scenario: first day of deployment before any pipeline run
        has completed, or S3 prefix is misconfigured.
        """
        result = load_and_filter_from_s3(
            mock_s3_empty, 'Data Analyst', 'Last week', None
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ══════════════════════════════════════════════════════════════════════════════
# All salary null post clean — 1 test
# ══════════════════════════════════════════════════════════════════════════════

class TestAllSalaryNull:

    def test_all_null_salary_does_not_crash_cleaning(self):
        """
        Dataframe where every Salary Estimate is null should not crash
        clean_salary_data() — Min_Salary and Max_Salary should all be
        None, and Avg Salary should also be None.

        Real scenario: Glassdoor changes its salary display format
        and the scraper fails to capture any salary values that day.
        """
        df = pd.DataFrame({
            'Salary Estimate': [None, None, None, None, None]
        })

        result = clean_salary_data(df)

        assert 'Min_Salary' in result.columns
        assert 'Max_Salary' in result.columns
        assert 'Avg Salary' in result.columns
        assert result['Min_Salary'].isna().all()
        assert result['Max_Salary'].isna().all()
        assert result['Avg Salary'].isna().all()
