#!/usr/bin/env python
# coding: utf-8

"""
Jobsdata_cleaning_final_version.py
────────────────────────────────────
Stage 2 of the pipeline — Data Cleaning.

Downloads the validated raw parquet from S3, runs all cleaning steps,
and uploads the cleaned file back to S3 under Cleaned_data/.

All core cleaning logic is unchanged from the original version.
Changes made:
    - Proper error handling around S3 download
    - Consistent file save path (all local files go to download_dir)
    - Full try/except wrapping so cron catches crashes
    - Crash alert fired via notifications.py on unexpected failure
    - Sensitive credentials pulled from environment variables
    - Groq LLM fallback added to extract_experience_years() for cases
      where regex returns None — JD is trimmed to requirements section
      first to minimise token usage

Required environment variables:
    GROQ_API_KEY — Groq API key from console.groq.com
"""

import os
import sys
import re
import time
import traceback
import pandas as pd
import numpy as np
import boto3
from groq import Groq
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from notifications import send_alert

pd.set_option('display.max_columns', None)


# ── Config ────────────────────────────────────────────────────────────────────
download_dir = '/home/ec2-user/data_downloads'
date         = datetime.now().strftime('%m-%d-%Y')
bucket       = 'recentjobslistingbucket'
LOG_FILE     = os.path.join(download_dir, f'Job_scrape_{date}.log')

os.makedirs(download_dir, exist_ok=True)

# ── Groq client ───────────────────────────────────────────────────────────────
# Set GROQ_API_KEY in /etc/environment or .bashrc on EC2
groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
GROQ_MODEL  = 'llama-3.1-8b-instant'   # fast, free tier, sufficient for extraction

# Section headers to scan for when trimming job descriptions
# Ordered by specificity — more specific headers are checked first
REQUIREMENTS_HEADERS = [
    'minimum requirements',
    'minimum qualifications',
    'required qualifications',
    'basic qualifications',
    'what you need',
    "what you'll need",
    'your background',
    'your experience',
    'experience required',
    'about you',
    'requirements',
    'qualifications',
]


# ── Cleaning functions (unchanged from original) ──────────────────────────────
def job_title_clean(x):
    if pd.isna(x):
        return 'Other'

    x = x.lower()

    if 'intern' in x or 'co-op' in x or 'co - op' in x:
        if 'ai/ml' in x or 'ml' in x or 'machine learning engineer' in x or 'learning' in x or 'mle' in x or 'ai' in x:
            return 'Machine Learning Intern'
        elif 'scientist' in x or 'data science' in x:
            return 'Data Scientist Intern'
        elif 'analyst' in x or 'analytics' in x or 'analysis' in x or 'business' in x or 'visualization' in x or 'data consultant' in x:
            return 'Data Analyst Intern'
        else:
            return 'Intern'

    if 'analyst' in x or 'analytics' in x or 'analysis' in x or 'business' in x or 'visualization' in x or 'data consultant' in x:
        return 'Data Analyst'
    elif 'scientist' in x or 'data science' in x:
        return 'Data Scientist'
    elif 'ai/ml' in x or 'ml' in x or 'machine learning engineer' in x or 'learning' in x or 'mle' in x:
        return 'Machine learning engineer'

    return 'Other'


def clean_salary_data(df):
    df = df.copy()

    salary = df['Salary Estimate'].apply(
        lambda x: x.split('/')[0].strip() if isinstance(x, str) and '/' in x else x
    )
    salary = salary.str.replace('$', '', regex=False).str.replace('K', '', regex=False)

    df['Hourly']  = df['Salary Estimate'].apply(lambda x: 1 if isinstance(x, str) and 'hr'  in x.lower() else 0)
    df['Monthly'] = df['Salary Estimate'].apply(lambda x: 1 if isinstance(x, str) and 'mo'  in x.lower() else 0)

    df['Min_Salary'] = salary.apply(
        lambda x: float(x.split('–')[0].strip()) if isinstance(x, str) and '–' in x
        else (float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else None)
    )
    df['Max_Salary'] = salary.apply(
        lambda x: float(x.split('–')[1].strip()) if isinstance(x, str) and '–' in x
        else (float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else None)
    )

    df['Min_Salary'] = df.apply(lambda x: x['Min_Salary'] * 2  if x['Hourly']  == 1 else x['Min_Salary'], axis=1)
    df['Max_Salary'] = df.apply(lambda x: x['Max_Salary'] * 2  if x['Hourly']  == 1 else x['Max_Salary'], axis=1)
    df['Min_Salary'] = df.apply(lambda x: x['Min_Salary'] * 12 if x['Monthly'] == 1 else x['Min_Salary'], axis=1)
    df['Max_Salary'] = df.apply(lambda x: x['Max_Salary'] * 12 if x['Monthly'] == 1 else x['Max_Salary'], axis=1)

    df['Avg Salary'] = (df['Min_Salary'] + df['Max_Salary']) / 2

    return df


def trim_to_requirements_section(text: str) -> str:
    """
    Trim a job description down to just the requirements section to
    reduce token input to Groq.

    Scans for known section headers (case-insensitive). Takes the earliest
    match and returns everything from that point onward.

    If no known header is found, returns the full text unchanged — the
    Groq call still happens, just with the full JD as input.
    """
    text_lower = text.lower()
    earliest_pos = len(text)

    for header in REQUIREMENTS_HEADERS:
        pos = text_lower.find(header)
        if pos != -1 and pos < earliest_pos:
            earliest_pos = pos

    if earliest_pos < len(text):
        trimmed = text[earliest_pos:]
        print(f'[trim] Requirements section found — trimmed from {len(text)} to {len(trimmed)} chars')
        return trimmed

    print(f'[trim] No requirements section found — using full JD ({len(text)} chars)')
    return text


def groq_experience_fallback(text: str) -> int | None:
    """
    Use Groq (llama-3.1-8b-instant) to extract minimum years of experience
    from a job description when regex returns None.

    The text passed in should already be trimmed to the requirements section
    by trim_to_requirements_section() where possible.

    Returns an integer (years) or None if extraction fails or is uncertain.
    """
    prompt = (
        "You are extracting the minimum years of experience required from a job description excerpt.\n"
        "Rules:\n"
        "  - Return ONLY a single integer — the minimum years of experience required.\n"
        "  - If the role is entry-level or explicitly requires no experience, return 0.\n"
        "  - If you cannot determine a specific number from the text, return -1.\n"
        "  - Do not return any other text, explanation, punctuation, or formatting.\n\n"
        f"Job description excerpt:\n{text}"
    )

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,       # deterministic — we want extraction not creativity
            max_tokens=5,        # we only need a single integer back
        )

        raw = response.choices[0].message.content.strip()

        # Parse response — expect a clean integer string
        value = int(raw)

        if value == -1:
            # Model signalled it couldn't determine — treat as None
            print(f'[groq] Model returned -1 — experience not determinable from text')
            return None

        print(f'[groq] Extracted experience via LLM: {value} years')
        return value

    except ValueError:
        # Model returned something non-numeric despite instructions
        print(f'[groq] Non-integer response received: "{raw}" — returning None')
        return None

    except Exception as e:
        # Rate limit or API error — log and return None gracefully
        # Pipeline continues, this record just stays None
        print(f'[groq] API call failed: {e} — returning None')
        time.sleep(2)   # brief pause before continuing if rate limited
        return None


def extract_experience_years(text: str) -> int | None:
    """
    Extract minimum years of experience from a job description.

    Strategy:
        1. Regex — handles explicit patterns instantly and for free
        2. Groq fallback — handles natural language that regex misses
           Input is trimmed to the requirements section before the API call
           to reduce token usage.
    """
    if pd.isna(text):
        return None

    text_lower = text.lower()

    # ── Regex layer ───────────────────────────────────────────────────────────
    match = re.search(
        r'(\d+)\s*(?:\+|\-to|-)?\s*(\d+)?\s*(?:years?|yrs?)\s+(?:of )?(?:relevant )?experience',
        text_lower
    )
    if match:
        return int(match.group(1))

    match = re.search(r'(?:at least|minimum|least|over)\s+(\d+)\s*(?:years?|yrs?)', text_lower)
    if match:
        return int(match.group(1))

    if re.search(r'(some|a few|couple of)\s+(years?\s+)?experience', text_lower):
        return 2

    if re.search(r'entry[-\s]?level|fresher|no (prior )?experience', text_lower):
        return 0

    # ── Groq fallback — only reached when regex returns nothing ───────────────
    print('[experience] Regex found nothing — falling back to Groq')
    trimmed_text = trim_to_requirements_section(text)
    return groq_experience_fallback(trimmed_text)


def seniority(title, experience):
    text = str(title).lower().strip()

    if any(keyword in text for keyword in ['intern', 'co-op', 'trainee', 'student']):
        return 'Intern'
    if any(keyword in text for keyword in ['senior', 'sr', 'lead', 'principal', 'manager', 'director', 'head']):
        return 'Senior'
    if any(keyword in text for keyword in ['junior', 'jr', 'jr.', 'entry-level', 'entry level']):
        return 'Junior'

    try:
        experience = int(experience)
        if experience >= 5:
            return 'Senior'
        elif experience <= 2:
            return 'Junior'
        elif 3 <= experience < 5:
            return 'Mid'
    except Exception:
        pass

    return np.nan


def education_extract(x):
    if pd.isna(x):
        return 'Not Mentioned'

    x = x.lower()

    if any(word in x for word in ['phd', 'ph.d', 'doctorate']):
        return 'PhD'
    elif any(word in x for word in ['master', 'm.s', 'ms', 'm.sc', 'mba']):
        return 'Masters'
    elif any(word in x for word in ['bachelor', 'b.s', 'b.sc', 'undergraduate']):
        return 'Bachelors'
    else:
        return 'Not Mentioned'


def extract_matches(text, skill_list):
    if not isinstance(text, str):
        return ['None']

    text_lower = text.lower()
    found = [
        skill for skill in skill_list
        if re.search(rf'\b{re.escape(skill.lower())}\b', text_lower)
    ]
    return found if found else ['None']


# ── Skill lists (unchanged from original) ─────────────────────────────────────
technologies = [
    'Python', 'SQL', 'R', 'Java', 'SAS', 'C', 'C++', 'JavaScript', 'TypeScript',
    'MATLAB', 'Julia', 'Go', 'Shell', 'Scala', 'Kotlin', 'Rust',
    'Excel', 'Jupyter', 'RStudio', 'NoSQL', 'PostgreSQL', 'MySQL', 'MongoDB'
]

operations = [
    'AWS', 'Azure', 'GCP', 'Git', 'Docker', 'Kubernetes', 'Jenkins', 'Airflow',
    'DBT', 'Snowflake', 'BigQuery', 'Redshift', 'Looker', 'PowerBI', 'Tableau',
    'CI/CD', 'DVC', 'VS Code', 'Apache Kafka', 'Apache Beam', 'Apache Flink',
    'Vertex AI', 'Weights & Biases', 'MLflow', 'Neptune', 'DataRobot', 'H2O.ai', 'AutoML'
]

packages = [
    'Pandas', 'NumPy', 'Scikit-learn', 'Statsmodels', 'XGBoost', 'LightGBM',
    'TensorFlow', 'Keras', 'PyTorch', 'Spark', 'Spark MLlib', 'Tidyverse', 'FastAPI',
    'OpenCV', 'Transformers', 'Hugging Face', 'NLTK', 'SpaCy', 'Gensim',
    'Plotly', 'Seaborn', 'Matplotlib', 'Altair', 'Dash', 'Bokeh',
    'BeautifulSoup', 'Scrapy', 'Prophet', 'CatBoost'
]


if __name__ == '__main__':
    s3 = boto3.client('s3', region_name='us-east-2')
    # ── Main ──────────────────────────────────────────────────────────────────────
    try:

        # ── Step 1: Download raw file from S3 ─────────────────────────────────────
        raw_file_name = f'final_jobs_df_raw_{date}.parquet'
        local_path    = os.path.join(download_dir, raw_file_name)
        s3_key        = f'Raw_data/{raw_file_name}'

        try:
            s3.download_file(bucket, s3_key, local_path)
            print(f'[download] Successfully downloaded {raw_file_name}')
        except Exception as e:
            raise RuntimeError(
                f'Failed to download raw file from S3 ({s3_key}). '
                f'Raw validation may not have completed. Error: {e}'
            )

        # ── Step 2: Load ──────────────────────────────────────────────────────────
        jobs_df = pd.read_parquet(local_path)
        print(f'[load] Raw dataset shape: {jobs_df.shape}')

        # ── Step 3: Deduplication ─────────────────────────────────────────────────
        before_dedup = jobs_df.shape[0]
        jobs_df      = jobs_df[~jobs_df.drop('Job_Role', axis=1).duplicated()]
        print(f'[dedup] Removed {before_dedup - jobs_df.shape[0]} duplicate rows')

        # ── Step 4: Clean job titles ──────────────────────────────────────────────
        jobs_df['Job_Title_Cleaned'] = jobs_df['Job Title'].apply(job_title_clean)
        jobs_df = jobs_df[~jobs_df['Job_Title_Cleaned'].isin(['Other', 'Intern'])]
        print(f'[title_clean] Shape after title filtering: {jobs_df.shape}')

        # ── Step 5: Clean salary ──────────────────────────────────────────────────
        jobs_df = clean_salary_data(jobs_df)
        print(f'[salary_clean] Salary columns parsed successfully')

        # ── Step 6: Extract state from location ───────────────────────────────────
        jobs_df['State'] = jobs_df['Location'].apply(
            lambda x: str(x).split(',')[1].strip() if ',' in str(x) else None
        )

        # ── Step 7: Extract experience ────────────────────────────────────────────
        jobs_df['Experience'] = jobs_df['Job Description'].apply(extract_experience_years)

        # ── Step 8: Seniority ─────────────────────────────────────────────────────
        jobs_df['Seniority_simplified'] = jobs_df.apply(
            lambda row: seniority(row['Job Title'], row['Experience']), axis=1
        )

        # ── Step 9: Education ─────────────────────────────────────────────────────
        jobs_df['Education'] = jobs_df['Job Description'].apply(education_extract)

        # ── Step 10: Skill extraction ─────────────────────────────────────────────
        jobs_df['Tech_Skills'] = jobs_df['Job Description'].apply(lambda x: extract_matches(x, technologies))
        jobs_df['Ops_Tools']   = jobs_df['Job Description'].apply(lambda x: extract_matches(x, operations))
        jobs_df['Libraries']   = jobs_df['Job Description'].apply(lambda x: extract_matches(x, packages))

        # ── Step 11: Binarize skills ──────────────────────────────────────────────
        mlb_tech = MultiLabelBinarizer()
        mlb_ops  = MultiLabelBinarizer()
        mlb_lib  = MultiLabelBinarizer()

        tech_df = pd.DataFrame(
            mlb_tech.fit_transform(jobs_df['Tech_Skills']),
            columns=[f'tech_{c}' for c in mlb_tech.classes_],
            index=jobs_df.index
        )
        ops_df = pd.DataFrame(
            mlb_ops.fit_transform(jobs_df['Ops_Tools']),
            columns=[f'ops_{c}' for c in mlb_ops.classes_],
            index=jobs_df.index
        )
        lib_df = pd.DataFrame(
            mlb_lib.fit_transform(jobs_df['Libraries']),
            columns=[f'lib_{c}' for c in mlb_lib.classes_],
            index=jobs_df.index
        )

        jobs_df = jobs_df.join([tech_df, ops_df, lib_df])

        # ── Step 12: Skill frequency summary ─────────────────────────────────────
        print('\n[skills] Top Tech Skills:')
        print(tech_df.sum().sort_values(ascending=False).head(10))
        print('\n[skills] Top Ops Tools:')
        print(ops_df.sum().sort_values(ascending=False).head(10))
        print('\n[skills] Top Libraries:')
        print(lib_df.sum().sort_values(ascending=False).head(10))

        # ── Step 13: Save cleaned file locally and upload to S3 ───────────────────
        cleaned_file_name = f'final_df_Cleaned_{date}.parquet'
        cleaned_local     = os.path.join(download_dir, cleaned_file_name)

        jobs_df.to_parquet(cleaned_local, index=False)
        print(f'[save] Saved cleaned file to {cleaned_local}')

        try:
            s3.upload_file(cleaned_local, bucket, f'Cleaned_data/{cleaned_file_name}')
            print(f'[s3] Successfully uploaded {cleaned_file_name} to Cleaned_data/')
        except Exception as e:
            raise RuntimeError(f'Failed to upload cleaned file to S3: {e}')

        # ── Step 14: Cleanup local files ──────────────────────────────────────────
        for f in os.listdir(download_dir):
            fp = os.path.join(download_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)
                print(f'[cleanup] Removed: {fp}')

        print(f'\n[cleaning] Completed. Final shape: {jobs_df.shape}')

    except SystemExit:
        raise

    except Exception as e:
        send_alert(
            stage='pipeline_crashed',
            details={
                'script':    'Jobsdata_cleaning_final_version.py',
                'error':     str(e),
                'traceback': traceback.format_exc()
            },
            log_file_path=LOG_FILE
        )
        sys.exit(1)
