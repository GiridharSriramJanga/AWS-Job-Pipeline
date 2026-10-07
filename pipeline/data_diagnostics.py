#!/usr/bin/env python
# coding: utf-8

"""
data_diagnostics.py
────────────────────
Stage 1 of the pipeline — Raw Data Validation.

Behaviour:
    FAIL → sends alert email, exits with code 1 (cron catches this)
    PASS → silent, uploads raw combined parquet to S3, cleans up local files

Imports send_alert from notifications.py — both files must be in the same directory.
"""

import os
import sys
import traceback
import boto3
import pandas as pd
from datetime import datetime
from notifications import send_alert

pd.set_option('display.max_columns', None)


# ── Config ────────────────────────────────────────────────────────────────────
save_dir   = '/home/ec2-user/data_downloads'
date       = datetime.now().strftime('%m-%d-%Y')
LOG_FILE   = os.path.join(save_dir, f'Job_scrape_{date}.log')

job_roles  = [
    'Data Analyst',
    'Data Scientist',
    'Machine learning engineer',
    'Data Analyst Intern',
    'Data Scientist Intern',
    'Machine Learning Intern',
]

# Thresholds
NULL_THRESHOLD      = 0.16   # max acceptable null rate per column
EMPTY_STR_THRESHOLD = 0.10   # max acceptable empty-string rate per column
MIN_ROWS_PER_ROLE   = 5      # minimum rows expected per job role
MIN_DESC_LENGTH     = 500    # minimum job description character length
ARTIFACT_THRESHOLD  = 0.16   # max acceptable scraping artifact rate
DUPLICATE_THRESHOLD = 0.16   # max acceptable exact duplicate rate

# Known UI artifacts that leak from the scraper
ARTIFACT_STRINGS    = ['show less', 'show more', 'see company reviews']


# ── File loading ──────────────────────────────────────────────────────────────
def load_raw_files():
    """
    Load each job role parquet file from save_dir.
    Returns the list of dataframes, roles found, and roles missing.
    """
    df_jobs       = []
    roles_found   = []
    roles_missing = []

    for role in job_roles:
        job_title = role.replace(' ', '_')
        file_name = f'{job_title}_jobs_{date}.parquet'
        file_path = os.path.join(save_dir, file_name)

        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            df['Job_Role'] = role
            df_jobs.append(df)
            roles_found.append(role)
            print(f'[load] Imported : {job_title}')
        else:
            roles_missing.append(role)
            print(f'[load] Not found: {job_title}')

    return df_jobs, roles_found, roles_missing


# ── Validation checks ─────────────────────────────────────────────────────────
def run_validation(df: pd.DataFrame, roles_found: list, roles_missing: list) -> dict:
    """
    Run all raw data quality checks.
    Returns a dict of {check_name: failure_reason}.
    Empty dict means all checks passed.
    """
    failed_checks = {}
    total_rows    = df.shape[0]

    # ── Check 1: Role coverage ────────────────────────────────────────────────
    if roles_missing:
        failed_checks['role_coverage'] = (
            f"The following roles had no scraped file today: "
            f"{', '.join(roles_missing)}"
        )

    # ── Check 2: Total row count ──────────────────────────────────────────────
    if total_rows == 0:
        failed_checks['total_row_count'] = (
            'DataFrame is completely empty — no data to validate or clean.'
        )
        # No point running further checks on an empty dataframe
        return failed_checks

    # ── Check 3: Per-role minimum row count ───────────────────────────────────
    role_counts = df['Job_Role'].value_counts()
    low_roles   = [
        f"{role} ({role_counts.get(role, 0)} rows)"
        for role in roles_found
        if role_counts.get(role, 0) < MIN_ROWS_PER_ROLE
    ]
    if low_roles:
        failed_checks['per_role_row_count'] = (
            f"Roles below minimum {MIN_ROWS_PER_ROLE} rows: {', '.join(low_roles)}"
        )

    # ── Check 4: Null threshold ───────────────────────────────────────────────
    null_problems = []
    for col in df.columns:
        null_pct = df[col].isna().sum() / total_rows
        if null_pct > NULL_THRESHOLD:
            null_problems.append(f'{col} ({null_pct:.1%} null)')
        else:
            print(f'[null_check] {col} — OK ({null_pct:.1%} null)')

    if null_problems:
        failed_checks['null_threshold'] = (
            f"Columns exceeding {NULL_THRESHOLD:.0%} null threshold: "
            f"{', '.join(null_problems)}"
        )

    # ── Check 5: Empty string check ───────────────────────────────────────────
    # Catches values that are not NaN but are blank — your current null
    # check would completely miss these
    empty_problems = []
    for col in df.select_dtypes(include='object').columns:
        empty_count = (df[col].astype(str).str.strip() == '').sum()
        empty_pct   = empty_count / total_rows
        if empty_pct > EMPTY_STR_THRESHOLD:
            empty_problems.append(f'{col} ({empty_pct:.1%} empty strings)')

    if empty_problems:
        failed_checks['empty_strings'] = (
            f"Columns with excessive empty strings (not null, but blank): "
            f"{', '.join(empty_problems)}"
        )

    # ── Check 6: Scraping artifact contamination ──────────────────────────────
    # UI button texts leaking into job descriptions from the scraper
    if 'Job Description' in df.columns:
        artifact_mask  = df['Job Description'].astype(str).str.lower().apply(
            lambda x: any(art in x for art in ARTIFACT_STRINGS)
        )
        artifact_count = artifact_mask.sum()
        artifact_pct   = artifact_count / total_rows

        print(f'[artifact_check] {artifact_count} records ({artifact_pct:.1%}) contain UI artifacts')

        if artifact_pct > ARTIFACT_THRESHOLD:
            failed_checks['scraping_artifacts'] = (
                f"{artifact_count} records ({artifact_pct:.1%}) contain scraping UI artifacts "
                f"('Show less', 'Show more', 'See company reviews') in Job Description. "
                f"Scraper HTML parsing may be broken."
            )

    # ── Check 7: Job description length ───────────────────────────────────────
    # Very short descriptions indicate a truncated or failed page scrape
    if 'Job Description' in df.columns:
        short_mask  = df['Job Description'].astype(str).str.len() < MIN_DESC_LENGTH
        short_count = short_mask.sum()
        short_pct   = short_count / total_rows

        if short_pct > NULL_THRESHOLD:
            failed_checks['desc_length'] = (
                f"{short_count} records ({short_pct:.1%}) have job descriptions "
                f"under {MIN_DESC_LENGTH} characters — likely truncated scrapes."
            )

    # ── Check 8: Exact duplicate rows ─────────────────────────────────────────
    dup_count = df.duplicated().sum()
    dup_pct   = dup_count / total_rows

    if dup_pct > DUPLICATE_THRESHOLD:
        failed_checks['duplicates'] = (
            f"{dup_count} exact duplicate rows ({dup_pct:.1%}) detected. "
            f"Scraper may have re-fetched the same listings multiple times."
        )

    return failed_checks


# ── S3 upload ─────────────────────────────────────────────────────────────────
def s3_file_upload(file_path: str, bucket_name: str, s3_bucket_path: str):
    s3   = boto3.client('s3', region_name='us-east-2')
    path = s3_bucket_path.rsplit('.', 1)[0]
    ext  = s3_bucket_path.rsplit('.', 1)[1]
    s3_key = f'{path}_{date}.{ext}'

    try:
        s3.upload_file(file_path, bucket_name, s3_key)
        print(f'[s3] Successfully uploaded to {s3_key}')
    except Exception as e:
        raise RuntimeError(f'S3 upload failed: {e}')


if __name__ == '__main__':
    # ── Main ──────────────────────────────────────────────────────────────────────
    try:

        # ── Step 1: Load files ────────────────────────────────────────────────────
        df_jobs, roles_found, roles_missing = load_raw_files()

        if not df_jobs:
            send_alert(
                stage='raw_failed',
                details={
                    'failed_checks': {
                        'file_loading': (
                            'No parquet files found for any role today. '
                            'The scraper may not have run or all files are missing.'
                        )
                    }
                },
                log_file_path=LOG_FILE
            )
            sys.exit(1)

        # ── Step 2: Combine ───────────────────────────────────────────────────────
        final_jobs_df = pd.concat(df_jobs, ignore_index=True)
        print(f'[combine] Total rows across all roles: {final_jobs_df.shape[0]}')

        # ── Step 3: Validate ──────────────────────────────────────────────────────
        failed_checks = run_validation(final_jobs_df, roles_found, roles_missing)

        if failed_checks:
            print(f'[validation] {len(failed_checks)} check(s) failed — aborting pipeline.')
            send_alert(
                stage='raw_failed',
                details={'failed_checks': failed_checks},
                log_file_path=LOG_FILE
            )
            sys.exit(1)

        print('[validation] All raw checks passed — proceeding to S3 upload.')

        # ── Step 4: Save and upload ───────────────────────────────────────────────
        output_path = os.path.join(save_dir, 'final_jobs_df_raw.parquet')
        final_jobs_df.to_parquet(output_path, index=False)
        s3_file_upload(output_path, 'recentjobslistingbucket', 'Raw_data/final_jobs_df_raw.parquet')

        # ── Step 5: Cleanup local files ───────────────────────────────────────────
        for f in os.listdir(save_dir):
            fp = os.path.join(save_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)
                print(f'[cleanup] Removed: {fp}')

    # Let sys.exit() pass through without being caught as a crash
    except SystemExit:
        raise

    # Catch any unexpected crash and alert
    except Exception as e:
        send_alert(
            stage='pipeline_crashed',
            details={
                'script':    'data_diagnostics.py',
                'error':     str(e),
                'traceback': traceback.format_exc()
            },
            log_file_path=LOG_FILE
        )
        sys.exit(1)
