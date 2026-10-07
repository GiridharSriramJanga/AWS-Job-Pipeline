"""
test_cleaning.py
─────────────────
Unit tests for all cleaning functions in
Jobsdata_cleaning_final_version.py.

Test count:
    job_title_clean              8
    clean_salary_data            6
    extract_experience_years     6
    seniority                    7
    education_extract            5
    trim_to_requirements_section 3
    ─────────────────────────────
    Total                       35
"""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# ── Set test env vars before any pipeline imports ─────────────────────────────
os.environ.setdefault('GROQ_API_KEY',         'test_groq_key')
os.environ.setdefault('ALERT_SENDER_MAIL',    'test@test.com')
os.environ.setdefault('ALERT_RECEIVER_MAIL',  'test@test.com')
os.environ.setdefault('GMAIL_APP_PASSWORD',   'test_password')

from pipeline.Jobsdata_cleaning_final_version import (
    job_title_clean,
    clean_salary_data,
    extract_experience_years,
    seniority,
    education_extract,
    trim_to_requirements_section,
)


# ══════════════════════════════════════════════════════════════════════════════
# job_title_clean — 8 tests
# ══════════════════════════════════════════════════════════════════════════════

class TestJobTitleClean:

    def test_data_analyst_title(self):
        """Standard data analyst title returns correct category."""
        assert job_title_clean('Senior Data Analyst') == 'Data Analyst'

    def test_data_scientist_title(self):
        """Data scientist variation returns correct category."""
        assert job_title_clean('Data Scientist II') == 'Data Scientist'

    def test_ml_engineer_title(self):
        """Machine learning engineer returns correct category."""
        assert job_title_clean('Machine Learning Engineer') == 'Machine learning engineer'

    def test_data_analyst_intern(self):
        """Intern with analyst keyword returns Data Analyst Intern."""
        assert job_title_clean('Data Analyst Intern') == 'Data Analyst Intern'

    def test_data_scientist_intern(self):
        """Intern with scientist keyword returns Data Scientist Intern."""
        assert job_title_clean('Data Science Intern') == 'Data Scientist Intern'

    def test_ml_intern(self):
        """Intern with ML keyword returns Machine Learning Intern."""
        assert job_title_clean('ML Intern') == 'Machine Learning Intern'

    def test_none_input_returns_other(self):
        """None input returns Other — no crash."""
        assert job_title_clean(None) == 'Other'

    def test_unrelated_title_returns_other(self):
        """Title unrelated to data roles returns Other."""
        assert job_title_clean('Quality Assurance Engineer') == 'Other'


# ══════════════════════════════════════════════════════════════════════════════
# clean_salary_data — 6 tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanSalaryData:

    def _make_df(self, salary_value):
        """Helper — returns a single-row dataframe with given salary."""
        return pd.DataFrame({'Salary Estimate': [salary_value]})

    def test_annual_salary_range_parsed(self):
        """Annual range '$80K – $100K/yr' produces correct Min and Max."""
        df     = self._make_df('$80K – $100K/yr (Employer provided)')
        result = clean_salary_data(df)
        assert result['Min_Salary'].iloc[0] == 80.0
        assert result['Max_Salary'].iloc[0] == 100.0

    def test_hourly_salary_converted_to_annual(self):
        """Hourly '$26 – $33/hr' is multiplied by 2 (×2 in your script)."""
        df     = self._make_df('$26 – $33/hr (Employer provided)')
        result = clean_salary_data(df)
        assert result['Hourly'].iloc[0] == 1
        assert result['Min_Salary'].iloc[0] == 52.0
        assert result['Max_Salary'].iloc[0] == 66.0

    def test_monthly_salary_converted_to_annual(self):
        """Monthly salary is multiplied by 12."""
        df     = self._make_df('$6K – $8K/mo (Employer provided)')
        result = clean_salary_data(df)
        assert result['Monthly'].iloc[0] == 1
        assert result['Min_Salary'].iloc[0] == 72.0
        assert result['Max_Salary'].iloc[0] == 96.0

    def test_single_value_salary(self):
        """Single salary value sets Min and Max to the same amount."""
        df     = self._make_df('$70K/yr (Employer provided)')
        result = clean_salary_data(df)
        assert result['Min_Salary'].iloc[0] == 70.0
        assert result['Max_Salary'].iloc[0] == 70.0

    def test_null_salary_returns_none(self):
        """Null salary produces None for Min, Max, and Avg."""
        df     = self._make_df(None)
        result = clean_salary_data(df)
        assert pd.isna(result['Min_Salary'].iloc[0])
        assert pd.isna(result['Max_Salary'].iloc[0])

    def test_avg_salary_computed_correctly(self):
        """Avg Salary equals (Min + Max) / 2."""
        df     = self._make_df('$60K – $100K/yr (Employer provided)')
        result = clean_salary_data(df)
        assert result['Avg Salary'].iloc[0] == 80.0


# ══════════════════════════════════════════════════════════════════════════════
# extract_experience_years — 6 tests
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractExperienceYears:

    def test_explicit_years_pattern(self):
        """'3+ years of experience' returns 3."""
        result = extract_experience_years(
            'We require 3+ years of experience in data analysis.'
        )
        assert result == 3

    def test_minimum_years_pattern(self):
        """'minimum 5 years' returns 5."""
        result = extract_experience_years(
            'Minimum 5 years of relevant experience required.'
        )
        assert result == 5

    def test_entry_level_returns_zero(self):
        """Entry-level description returns 0."""
        result = extract_experience_years(
            'This is an entry-level position. No prior experience needed.'
        )
        assert result == 0

    def test_vague_experience_returns_two(self):
        """'a few years experience' returns hardcoded 2."""
        result = extract_experience_years(
            'Candidates should have a few years of experience in analytics.'
        )
        assert result == 2

    def test_none_input_returns_none(self):
        """None input returns None without crashing."""
        assert extract_experience_years(None) is None

    @patch('pipeline.Jobsdata_cleaning_final_version.groq_client')
    def test_groq_fallback_called_when_regex_fails(self, mock_groq):
        """
        When regex finds nothing, Groq fallback is triggered.
        Groq returns '4' → function returns 4.
        """
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '4'
        mock_groq.chat.completions.create.return_value = mock_response

        result = extract_experience_years(
            'We expect candidates to have spent meaningful time '
            'building and maintaining production data pipelines.'
        )

        assert result == 4
        mock_groq.chat.completions.create.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# seniority — 7 tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSeniority:

    def test_senior_keyword_in_title(self):
        """'Senior' in title returns Senior regardless of experience."""
        result = seniority('Senior Data Analyst', 2)
        assert result == 'Senior'

    def test_junior_keyword_in_title(self):
        """'Junior' in title returns Junior."""
        result = seniority('Junior Data Analyst', 5)
        assert result == 'Junior'

    def test_intern_keyword_in_title(self):
        """'Intern' in title always returns Intern."""
        result = seniority('Data Analyst Intern', 0)
        assert result == 'Intern'

    def test_experience_five_plus_returns_senior(self):
        """No title keyword but experience >= 5 returns Senior."""
        result = seniority('Data Analyst', 5)
        assert result == 'Senior'

    def test_experience_two_or_less_returns_junior(self):
        """No title keyword but experience <= 2 returns Junior."""
        result = seniority('Data Analyst', 2)
        assert result == 'Junior'

    def test_experience_three_to_four_returns_mid(self):
        """Experience 3-4 returns Mid."""
        result = seniority('Data Analyst', 3)
        assert result == 'Mid'

    def test_no_keywords_no_experience_returns_nan(self):
        """No title keywords and no parseable experience returns NaN."""
        result = seniority('Data Analyst', None)
        assert result is np.nan or (isinstance(result, float) and np.isnan(result))


# ══════════════════════════════════════════════════════════════════════════════
# education_extract — 5 tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEducationExtract:

    def test_bachelor_detected(self):
        """Bachelor keyword returns Bachelors."""
        result = education_extract(
            "Bachelor's degree in Computer Science required."
        )
        assert result == 'Bachelors'

    def test_masters_detected(self):
        """Master keyword returns Masters."""
        result = education_extract(
            "Master's degree or equivalent experience preferred."
        )
        assert result == 'Masters'

    def test_phd_detected(self):
        """PhD keyword returns PhD."""
        result = education_extract(
            'PhD in Statistics or related field required.'
        )
        assert result == 'PhD'

    def test_no_education_mentioned(self):
        """No education keywords returns Not Mentioned."""
        result = education_extract(
            'Strong analytical and communication skills required.'
        )
        assert result == 'Not Mentioned'

    def test_none_input_returns_not_mentioned(self):
        """None input returns Not Mentioned without crashing."""
        assert education_extract(None) == 'Not Mentioned'


# ══════════════════════════════════════════════════════════════════════════════
# trim_to_requirements_section — 3 tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTrimToRequirementsSection:

    def test_requirements_header_found_and_trimmed(self):
        """Text before 'requirements' section is trimmed correctly."""
        jd = (
            "About Us: We are a great company. We love data.\n"
            "What We Offer: Competitive salary and benefits.\n"
            "Requirements: 3+ years Python. SQL required. AWS preferred."
        )
        result = trim_to_requirements_section(jd)
        assert result.startswith('Requirements:')
        assert 'About Us' not in result

    def test_minimum_qualifications_header_found(self):
        """'Minimum qualifications' header is correctly identified."""
        jd = (
            "Job Overview: Great role at a great company.\n"
            "Minimum Qualifications: Bachelor degree. 2+ years experience."
        )
        result = trim_to_requirements_section(jd)
        assert 'Job Overview' not in result
        assert 'Minimum Qualifications' in result

    def test_no_section_found_returns_full_text(self):
        """When no known header is found, full text is returned unchanged."""
        jd = (
            "We are hiring a data analyst. "
            "Please send your resume to hr@company.com."
        )
        result = trim_to_requirements_section(jd)
        assert result == jd
