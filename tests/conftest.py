"""
conftest.py
────────────
Shared pytest fixtures used across all test files.
Loaded automatically by pytest — no imports needed in test files.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from io import BytesIO


# ── Raw dataframe fixtures ────────────────────────────────────────────────────

@pytest.fixture
def sample_raw_df():
    """
    Minimal valid raw dataframe as it would arrive from the scraper.
    Covers all 6 job roles with enough rows to pass validation thresholds.
    """
    roles = [
        'Data Analyst', 'Data Scientist', 'Machine learning engineer',
        'Data Analyst Intern', 'Data Scientist Intern', 'Machine Learning Intern'
    ]
    rows = []
    for role in roles:
        for i in range(8):  # 8 rows per role — above MIN_ROWS_PER_ROLE threshold
            rows.append({
                'Company Name'    : f'Company {i}',
                'Job Title'       : f'{role} {i}',
                'Location'        : f'Chicago, IL',
                'Job Description' : (
                    f'We are looking for a {role}. '
                    f'Requirements: 3+ years of experience in data analysis. '
                    f'Proficiency in Python and SQL required. '
                    f'Bachelor degree preferred. ' * 30   # long enough to pass length check
                ),
                'Salary Estimate' : '$80K – $100K/yr (Employer provided)',
                'Scrapped_date'   : '11-11-2025',
                'Job_Role'        : role,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_raw_df_with_problems():
    """Raw dataframe with intentional issues for negative testing."""
    return pd.DataFrame({
        'Company Name'    : ['Company A', None, 'Company C', 'Company D'],
        'Job Title'       : ['Data Analyst', 'Data Scientist', None, 'Data Analyst'],
        'Location'        : ['Chicago, IL', '', 'New York, NY', 'Austin, TX'],
        'Job Description' : [
            'Short',                           # too short
            'Normal description. ' * 50,
            'See company reviews Show less',   # artifact
            'Normal description. ' * 50,
        ],
        'Salary Estimate' : ['$80K/yr', '$60K – $70K/yr', None, None],
        'Scrapped_date'   : ['11-11-2025'] * 4,
        'Job_Role'        : ['Data Analyst'] * 4,
    })


@pytest.fixture
def sample_cleaned_df():
    """
    Minimal valid cleaned dataframe — all expected columns present
    and populated. Used for post-clean validation tests.
    """
    return pd.DataFrame({
        'Company Name'         : ['Company A', 'Company B', 'Company C',
                                  'Company D', 'Company E', 'Company F',
                                  'Company G', 'Company H', 'Company I', 'Company J'],
        'Job Title'            : ['Data Analyst'] * 5 + ['Data Scientist'] * 5,
        'Location'             : ['Chicago, IL'] * 10,
        'Job Description'      : ['Python SQL required. 3+ years experience. '
                                  'Bachelor degree. AWS Tableau.' ] * 10,
        'Salary Estimate'      : ['$80K – $100K/yr (Employer provided)'] * 10,
        'Scrapped_date'        : ['11-11-2025'] * 10,
        'Job_Role'             : ['Data Analyst'] * 5 + ['Data Scientist'] * 5,
        'Job_Title_Cleaned'    : ['Data Analyst'] * 5 + ['Data Scientist'] * 5,
        'Hourly'               : [0] * 10,
        'Monthly'              : [0] * 10,
        'Min_Salary'           : [80.0] * 10,
        'Max_Salary'           : [100.0] * 10,
        'Avg Salary'           : [90.0] * 10,
        'State'                : ['IL'] * 10,
        'Experience'           : [3] * 10,
        'Seniority_simplified' : ['Mid'] * 10,
        'Education'            : ['Bachelors'] * 10,
        'Tech_Skills'          : [['Python', 'SQL']] * 10,
        'Ops_Tools'            : [['AWS', 'Tableau']] * 10,
        'Libraries'            : [['Pandas']] * 10,
        'tech_Python'          : [1] * 10,
        'tech_SQL'             : [1] * 10,
        'ops_AWS'              : [1] * 10,
        'ops_Tableau'          : [1] * 10,
        'lib_Pandas'           : [1] * 10,
    })


# ── Resume fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_resume_text():
    return (
        "John Doe | Data Analyst\n"
        "Experience: 3 years of experience in data analysis.\n"
        "Skills: Python, SQL, Tableau, Excel, Pandas, NumPy.\n"
        "Worked with AWS for cloud storage and ETL pipelines.\n"
        "Education: Bachelor of Science in Statistics.\n"
        "Previous roles included building dashboards in Tableau "
        "and writing complex SQL queries for reporting."
    )


@pytest.fixture
def empty_resume_text():
    return ""


@pytest.fixture
def no_skills_resume_text():
    return (
        "Experienced professional with a background in business operations. "
        "Strong communication and leadership skills. "
        "Managed teams and delivered projects on time."
    )


# ── Embedding fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def mock_resume_embeddings():
    """Normalised random embeddings simulating encoded resume chunks."""
    np.random.seed(42)
    emb = np.random.randn(3, 384).astype(np.float32)
    # L2 normalise — embeddings.py uses normalize_embeddings=True
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


@pytest.fixture
def mock_job_embeddings_similar(mock_resume_embeddings):
    """Job embeddings very similar to resume — high cosine similarity."""
    noise = np.random.randn(*mock_resume_embeddings.shape).astype(np.float32) * 0.05
    emb   = mock_resume_embeddings + noise
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return (emb / norms).tolist()


@pytest.fixture
def mock_job_embeddings_dissimilar():
    """Job embeddings completely different from resume — low cosine similarity."""
    np.random.seed(99)
    emb   = np.random.randn(4, 384).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return (emb / norms).tolist()


@pytest.fixture
def mock_job_embeddings_empty():
    return []


# ── S3 / AWS mock helpers ─────────────────────────────────────────────────────

@pytest.fixture
def mock_s3_client():
    """Generic mocked boto3 S3 client."""
    return MagicMock()


@pytest.fixture
def mock_s3_with_parquet(sample_cleaned_df):
    """
    Mocked S3 client that returns a valid parquet file when download_fileobj
    is called. Used for integration tests of load_and_filter_from_s3().
    """
    buffer = BytesIO()
    sample_cleaned_df['chunk_embeddings'] = [
        [[0.1] * 384] * 3 for _ in range(len(sample_cleaned_df))
    ]
    sample_cleaned_df.to_parquet(buffer)
    buffer.seek(0)
    parquet_bytes = buffer.read()

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


@pytest.fixture
def mock_s3_empty():
    """Mocked S3 client that returns no files."""
    mock_client = MagicMock()
    mock_client.list_objects_v2.return_value = {}
    return mock_client
