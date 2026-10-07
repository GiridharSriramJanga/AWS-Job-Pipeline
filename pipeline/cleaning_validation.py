#!/usr/bin/env python
# coding: utf-8

"""
cleaning_validation.py
───────────────────────
Stage 3 of the pipeline — Post-Clean Data Validation.

Downloads the cleaned parquet from S3 and validates it.

Behaviour:
    PASS → sends success email (only email the pipeline sends on a healthy run)
    FAIL → sends alert email with exactly which checks failed and why

This is the last gate before data is considered production-ready.
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
download_dir   = '/home/ec2-user/data_downloads'
date           = datetime.now().strftime('%m-%d-%Y')
bucket         = 'recentjobslistingbucket'
LOG_FILE       = os.path.join(download_dir, f'Job_scrape_{date}.log')

os.makedirs(download_dir, exist_ok=True)

# Thresholds — slightly more lenient than raw since cleaning
# intentionally removes rows and transforms values
NULL_THRESHOLD      = 0.20
SENIORITY_THRESHOLD = 0.50   # seniority is harder to extract so given more room
MIN_ROWS            = 10     # absolute minimum after cleaning

# All columns that must exist in a properly cleaned dataset
EXPECTED_COLUMNS = [
    'Company Name', 'Job Title', 'Location', 'Job Description',
    'Salary Estimate', 'Scrapped_date', 'Job_Role',
    'Job_Title_Cleaned', 'Hourly', 'Monthly',
    'Min_Salary', 'Max_Salary', 'Avg Salary',
    'State', 'Experience', 'Seniority_simplified', 'Education',
    'Tech_Skills', 'Ops_Tools', 'Libraries'
]


# ── Validation checks ─────────────────────────────────────────────────────────
def run_post_clean_validation(df: pd.DataFrame) -> dict:
    """
    Validate the cleaned dataset.
    Returns a dict of {check_name: failure_reason}.
    Empty dict means all checks passed.
    """
    failed_checks = {}
    total_rows    = df.shape[0]

    # ── Check 1: Minimum row count after cleaning ──────────────────────────────
    # Catches cases where cleaning filters removed too aggressively
    if total_rows < MIN_ROWS:
        failed_checks['row_count_post_clean'] = (
            f'Only {total_rows} rows remain after cleaning — '
            f'expected at least {MIN_ROWS}. '
            f'Title filtering or deduplication may be too aggressive.'
        )
        # Not safe to run further checks with near-empty data
        return failed_checks

    # ── Check 2: Expected columns present ─────────────────────────────────────
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        failed_checks['missing_columns'] = (
            f'The following expected columns are missing after cleaning: '
            f"{', '.join(missing_cols)}"
        )

    # ── Check 3: No residual 'Other' or 'Intern' in Job_Title_Cleaned ─────────
    if 'Job_Title_Cleaned' in df.columns:
        bad_count = df['Job_Title_Cleaned'].isin(['Other', 'Intern']).sum()
        if bad_count > 0:
            failed_checks['job_title_cleaned'] = (
                f"{bad_count} records still contain 'Other' or 'Intern' in "
                f"Job_Title_Cleaned — the title filter in cleaning did not run correctly."
            )

    # ── Check 4: Salary null rate ──────────────────────────────────────────────
    for sal_col in ['Min_Salary', 'Max_Salary', 'Avg Salary']:
        if sal_col in df.columns:
            null_pct = df[sal_col].isna().sum() / total_rows
            if null_pct > NULL_THRESHOLD:
                failed_checks[f'{sal_col}_nulls'] = (
                    f'{null_pct:.1%} of {sal_col} is null — '
                    f'salary parsing in clean_salary_data() may have failed.'
                )
            else:
                print(f'[salary_check] {sal_col} — OK ({null_pct:.1%} null)')

    # ── Check 5: Salary range validity ────────────────────────────────────────
    # Min should never be greater than Max — indicates a parsing bug
    if 'Min_Salary' in df.columns and 'Max_Salary' in df.columns:
        both_present = df[['Min_Salary', 'Max_Salary']].dropna()
        inverted     = (both_present['Min_Salary'] > both_present['Max_Salary']).sum()
        if inverted > 0:
            failed_checks['salary_range_validity'] = (
                f'{inverted} records where Min_Salary > Max_Salary — '
                f'salary range parsing produced inverted values.'
            )

    # ── Check 6: Salary floor sanity check ────────────────────────────────────
    # Flag suspiciously low annual salaries for data/ML roles
    if 'Min_Salary' in df.columns:
        hourly_flag = df.get('Hourly', pd.Series([0] * total_rows))
        annual_mask = (hourly_flag == 0)
        low_salary  = (df.loc[annual_mask, 'Min_Salary'] < 20).sum()
        if low_salary > 0:
            failed_checks['salary_floor'] = (
                f'{low_salary} annual salary records have Min_Salary below $20K — '
                f'likely a unit conversion error (e.g., hourly not converted correctly).'
            )

    # ── Check 7: State null rate ───────────────────────────────────────────────
    if 'State' in df.columns:
        null_pct = df['State'].isna().sum() / total_rows
        if null_pct > NULL_THRESHOLD:
            failed_checks['state_nulls'] = (
                f'{null_pct:.1%} of State column is null — '
                f'location parsing may have failed for most records.'
            )
        else:
            print(f'[state_check] State — OK ({null_pct:.1%} null)')

    # ── Check 8: Seniority null rate ───────────────────────────────────────────
    if 'Seniority_simplified' in df.columns:
        null_pct = df['Seniority_simplified'].isna().sum() / total_rows
        if null_pct > SENIORITY_THRESHOLD:
            failed_checks['seniority_nulls'] = (
                f'{null_pct:.1%} of Seniority_simplified is null — '
                f'seniority extraction may be failing across most records.'
            )
        else:
            print(f'[seniority_check] Seniority_simplified — OK ({null_pct:.1%} null)')

    # ── Check 9: Education not returning all 'Not Mentioned' ──────────────────
    if 'Education' in df.columns:
        not_mentioned_pct = (df['Education'] == 'Not Mentioned').sum() / total_rows
        if not_mentioned_pct > 0.90:
            failed_checks['education_extraction'] = (
                f'{not_mentioned_pct:.1%} of Education is "Not Mentioned" — '
                f'education extraction keywords may not be matching job descriptions.'
            )
        else:
            print(f'[education_check] Education — OK ({not_mentioned_pct:.1%} Not Mentioned)')

    # ── Check 10: Skill columns are not all 'None' ─────────────────────────────
    for skill_col in ['Tech_Skills', 'Ops_Tools', 'Libraries']:
        if skill_col in df.columns:
            all_none_pct = df[skill_col].apply(
                lambda x: x == ['None'] if isinstance(x, list) else str(x) == 'None'
            ).sum() / total_rows
            if all_none_pct > 0.90:
                failed_checks[f'{skill_col}_extraction'] = (
                    f'{all_none_pct:.1%} of {skill_col} returned no matches — '
                    f'skill extraction regex may not be finding anything in descriptions.'
                )
            else:
                print(f'[skill_check] {skill_col} — OK ({all_none_pct:.1%} returned no matches)')

    return failed_checks


if __name__ == '__main__':
    s3 = boto3.client('s3', region_name='us-east-2')
    # ── Main ──────────────────────────────────────────────────────────────────────
    try:

        # ── Step 1: Download cleaned file from S3 ─────────────────────────────────
        cleaned_file_name = f'final_df_Cleaned_{date}.parquet'
        local_path        = os.path.join(download_dir, cleaned_file_name)
        s3_key            = f'Cleaned_data/{cleaned_file_name}'

        try:
            s3.download_file(bucket, s3_key, local_path)
            print(f'[download] Successfully downloaded {cleaned_file_name}')
        except Exception as e:
            raise RuntimeError(
                f'Failed to download cleaned file from S3 ({s3_key}). '
                f'Cleaning script may not have completed successfully. Error: {e}'
            )

        # ── Step 2: Load ──────────────────────────────────────────────────────────
        jobs_df = pd.read_parquet(local_path)
        print(f'[load] Cleaned dataset shape: {jobs_df.shape}')

        # ── Step 3: Validate ──────────────────────────────────────────────────────
        failed_checks = run_post_clean_validation(jobs_df)

        if failed_checks:
            print(f'[validation] {len(failed_checks)} post-clean check(s) failed.')
            send_alert(
                stage='clean_failed',
                details={'failed_checks': failed_checks},
                log_file_path=LOG_FILE
            )
            sys.exit(1)

        # ── Step 4: All checks passed — send success email ─────────────────────────
        print('[validation] All post-clean checks passed.')

        roles_present = (
            ', '.join(sorted(jobs_df['Job_Role'].unique().tolist()))
            if 'Job_Role' in jobs_df.columns else 'N/A'
        )

        send_alert(
            stage='pipeline_success',
            details={
                'row_count':     jobs_df.shape[0],
                'roles_present': roles_present
            },
            log_file_path=LOG_FILE
        )

        # ── Step 5: Cleanup local file ────────────────────────────────────────────
        if os.path.exists(local_path):
            os.remove(local_path)
            print(f'[cleanup] Removed: {local_path}')

    except SystemExit:
        raise

    except Exception as e:
        send_alert(
            stage='pipeline_crashed',
            details={
                'script':    'cleaning_validation.py',
                'error':     str(e),
                'traceback': traceback.format_exc()
            },
            log_file_path=LOG_FILE
        )
        sys.exit(1)
