"""
test_validation.py
───────────────────
Unit tests for validation logic in
data_diagnostics.py and cleaning_validation.py.

Test count:
    run_validation             8
    run_post_clean_validation  9
    ───────────────────────────
    Total                     17
"""

import os
import pytest
import numpy as np
import pandas as pd

# ── Set test env vars before any pipeline imports ─────────────────────────────
os.environ.setdefault('GROQ_API_KEY',        'test_groq_key')
os.environ.setdefault('ALERT_SENDER_MAIL',   'test@test.com')
os.environ.setdefault('ALERT_RECEIVER_MAIL', 'test@test.com')
os.environ.setdefault('GMAIL_APP_PASSWORD',  'test_password')

from pipeline.data_diagnostics import run_validation
from pipeline.cleaning_validation import run_post_clean_validation


# ══════════════════════════════════════════════════════════════════════════════
# run_validation — 8 tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRunValidation:

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_valid_df(self, n_per_role=8):
        """
        Build a valid raw dataframe covering all 6 roles with enough
        rows and clean data to pass every validation check.
        """
        job_roles = [
            'Data Analyst', 'Data Scientist', 'Machine learning engineer',
            'Data Analyst Intern', 'Data Scientist Intern', 'Machine Learning Intern'
        ]
        rows = []
        for role in job_roles:
            for i in range(n_per_role):
                rows.append({
                    'Company Name'    : f'Company {i}',
                    'Job Title'       : f'{role} {i}',
                    'Location'        : f'Chicago, IL',
                    'Job Description' : 'Normal job description. ' * 40,
                    'Salary Estimate' : '$80K – $100K/yr (Employer provided)',
                    'Scrapped_date'   : '11-11-2025',
                    'Job_Role'        : role,
                })
        return pd.DataFrame(rows)

    # ── Tests ─────────────────────────────────────────────────────────────────

    def test_valid_df_passes_all_checks(self, sample_raw_df):
        """Clean raw df with all roles present passes all checks."""
        roles_found   = [
            'Data Analyst', 'Data Scientist', 'Machine learning engineer',
            'Data Analyst Intern', 'Data Scientist Intern', 'Machine Learning Intern'
        ]
        roles_missing = []
        result        = run_validation(sample_raw_df, roles_found, roles_missing)
        assert result == {}

    def test_missing_roles_flagged(self, sample_raw_df):
        """Missing roles appear in failed_checks under role_coverage."""
        roles_found   = ['Data Analyst']
        roles_missing = ['Data Scientist', 'Machine learning engineer']
        result        = run_validation(sample_raw_df, roles_found, roles_missing)
        assert 'role_coverage' in result
        assert 'Data Scientist' in result['role_coverage']

    def test_null_threshold_exceeded(self):
        """Column with >16% nulls is flagged under null_threshold."""
        df = self._make_valid_df()
        # Set 30% of Salary Estimate to null — above 16% threshold
        null_count = int(len(df) * 0.30)
        df.loc[:null_count, 'Salary Estimate'] = None

        result = run_validation(df, list(df['Job_Role'].unique()), [])
        assert 'null_threshold' in result
        assert 'Salary Estimate' in result['null_threshold']

    def test_empty_string_detection(self):
        """Column with excessive empty strings is flagged — not nulls, empty strings."""
        df = self._make_valid_df()
        # Set 20% of Location to empty string
        empty_count = int(len(df) * 0.20)
        df.loc[:empty_count, 'Location'] = ''

        result = run_validation(df, list(df['Job_Role'].unique()), [])
        assert 'empty_strings' in result

    def test_scraping_artifacts_detected(self):
        """Descriptions containing UI artifact text are flagged."""
        df = self._make_valid_df()
        # Inject artifacts into 30% of descriptions — above threshold
        artifact_count = int(len(df) * 0.30)
        df.loc[:artifact_count, 'Job Description'] = (
            'Normal description content. Show less See company reviews'
        )
        result = run_validation(df, list(df['Job_Role'].unique()), [])
        assert 'scraping_artifacts' in result

    def test_short_descriptions_flagged(self):
        """Descriptions under 500 chars in >16% of rows are flagged."""
        df = self._make_valid_df()
        short_count = int(len(df) * 0.20)
        df.loc[:short_count, 'Job Description'] = 'Short.'

        result = run_validation(df, list(df['Job_Role'].unique()), [])
        assert 'desc_length' in result

    def test_high_duplicate_rate_flagged(self):
        """DataFrames with >16% exact duplicates are flagged."""
        base_df  = self._make_valid_df(n_per_role=5)
        # Duplicate 30% of rows
        dup_df   = base_df.iloc[:int(len(base_df) * 0.30)]
        combined = pd.concat([base_df, dup_df], ignore_index=True)

        result = run_validation(combined, list(combined['Job_Role'].unique()), [])
        assert 'duplicates' in result

    def test_empty_dataframe_returns_row_count_failure(self):
        """Completely empty df fails immediately on total_row_count."""
        result = run_validation(pd.DataFrame(), [], [])
        assert 'total_row_count' in result


# ══════════════════════════════════════════════════════════════════════════════
# run_post_clean_validation — 9 tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRunPostCleanValidation:

    def test_valid_cleaned_df_passes_all_checks(self, sample_cleaned_df):
        """Valid cleaned df with all expected columns passes all checks."""
        result = run_post_clean_validation(sample_cleaned_df)
        assert result == {}

    def test_other_in_job_title_cleaned_flagged(self, sample_cleaned_df):
        """Residual 'Other' values in Job_Title_Cleaned are flagged."""
        df                       = sample_cleaned_df.copy()
        df.loc[0, 'Job_Title_Cleaned'] = 'Other'
        result                   = run_post_clean_validation(df)
        assert 'job_title_cleaned' in result

    def test_high_salary_null_rate_flagged(self, sample_cleaned_df):
        """Min_Salary with >20% nulls is flagged."""
        df = sample_cleaned_df.copy()
        null_count = int(len(df) * 0.25)
        df.loc[:null_count, 'Min_Salary'] = None
        result = run_post_clean_validation(df)
        assert 'Min_Salary_nulls' in result

    def test_inverted_salary_range_flagged(self, sample_cleaned_df):
        """Min_Salary > Max_Salary is flagged as salary_range_validity."""
        df = sample_cleaned_df.copy()
        df.loc[0, 'Min_Salary'] = 200.0
        df.loc[0, 'Max_Salary'] = 50.0
        result = run_post_clean_validation(df)
        assert 'salary_range_validity' in result

    def test_suspiciously_low_salary_flagged(self, sample_cleaned_df):
        """Annual salary below $20K is flagged as likely conversion error."""
        df = sample_cleaned_df.copy()
        df['Min_Salary'] = 5.0   # $5K annual — clearly wrong
        df['Hourly']     = 0     # mark as annual, not hourly
        result = run_post_clean_validation(df)
        assert 'salary_floor' in result

    def test_high_state_null_rate_flagged(self, sample_cleaned_df):
        """State column with >20% nulls is flagged."""
        df = sample_cleaned_df.copy()
        null_count = int(len(df) * 0.25)
        df.loc[:null_count, 'State'] = None
        result = run_post_clean_validation(df)
        assert 'state_nulls' in result

    def test_missing_expected_columns_flagged(self, sample_cleaned_df):
        """Missing expected columns are flagged under missing_columns."""
        df     = sample_cleaned_df.drop(columns=['Seniority_simplified', 'Education'])
        result = run_post_clean_validation(df)
        assert 'missing_columns' in result
        assert 'Seniority_simplified' in result['missing_columns']

    def test_all_education_not_mentioned_flagged(self, sample_cleaned_df):
        """When >90% of Education is 'Not Mentioned', extraction is flagged."""
        df              = sample_cleaned_df.copy()
        df['Education'] = 'Not Mentioned'
        result          = run_post_clean_validation(df)
        assert 'education_extraction' in result

    def test_row_count_below_minimum_flagged(self, sample_cleaned_df):
        """Less than MIN_ROWS (10) rows after cleaning triggers check."""
        df     = sample_cleaned_df.iloc[:3].copy()   # only 3 rows
        result = run_post_clean_validation(df)
        assert 'row_count_post_clean' in result
