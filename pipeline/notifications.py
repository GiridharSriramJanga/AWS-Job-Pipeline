#!/usr/bin/env python
# coding: utf-8

"""
notifications.py
────────────────
Shared alert utility imported by data_diagnostics.py and cleaning_validation.py.
Never call this script directly.

Required environment variables (set in /etc/environment or .bashrc on EC2):
    ALERT_SENDER_MAIL   — Gmail address sending the alert
    ALERT_RECEIVER_MAIL — Gmail address receiving the alert
    GMAIL_APP_PASSWORD  — Gmail app password (16-char, no spaces)
"""

import os
import smtplib
from email.message import EmailMessage
from datetime import datetime


def send_alert(stage: str, details: dict, log_file_path: str = None):
    """
    Unified notification dispatcher for all pipeline stages.

    Parameters
    ----------
    stage : str
        One of:
            'raw_failed'        — raw validation failed, pipeline aborted
            'clean_failed'      — post-clean validation failed
            'pipeline_success'  — full pipeline completed successfully
            'pipeline_crashed'  — unexpected exception anywhere in pipeline
    details : dict
        Context for the email body. Keys depend on stage:
            raw_failed / clean_failed → {'failed_checks': {check_name: reason}}
            pipeline_success          → {'row_count': int, 'roles_present': str}
            pipeline_crashed          → {'script': str, 'error': str, 'traceback': str}
    log_file_path : str, optional
        Path to a .log file to attach to the email if it exists.
    """

    sender_mail    = os.environ.get('ALERT_SENDER_MAIL')
    receiver_mail  = os.environ.get('ALERT_RECEIVER_MAIL')
    email_password = os.environ.get('GMAIL_APP_PASSWORD')

    if not all([sender_mail, receiver_mail, email_password]):
        raise EnvironmentError(
            "Missing one or more required environment variables: "
            "ALERT_SENDER_MAIL, ALERT_RECEIVER_MAIL, GMAIL_APP_PASSWORD"
        )

    timestamp = datetime.now().strftime("%m-%d-%Y %H:%M:%S")

    subjects = {
        'raw_failed':       '[PIPELINE ABORTED] Raw Validation Failed',
        'clean_failed':     '[PIPELINE WARNING] Post-Clean Validation Failed',
        'pipeline_success': '[PIPELINE SUCCESS] Scraper & Cleaning Good',
        'pipeline_crashed': '[PIPELINE CRASHED] Unexpected Error Occurred',
    }

    subject = subjects.get(stage, '[PIPELINE ALERT] Unknown Stage')

    # ── Build email body ──────────────────────────────────────────────────────
    divider = '─' * 52
    lines   = [
        'Scraper Pipeline Notification',
        divider,
        f'Timestamp : {timestamp}',
        f'Stage     : {stage.replace("_", " ").upper()}',
        divider,
        '',
    ]

    if stage == 'pipeline_success':
        lines += [
            '  Raw validation      PASSED',
            '  Cleaning            COMPLETED',
            '  Post-clean validation PASSED',
            '  S3 upload           COMPLETED',
            '',
            divider,
            f"  Total rows in cleaned dataset : {details.get('row_count', 'N/A')}",
            f"  Roles present                 : {details.get('roles_present', 'N/A')}",
        ]

    elif stage in ('raw_failed', 'clean_failed'):
        if stage == 'raw_failed':
            lines += [
                '  Raw validation FAILED.',
                '  Cleaning script was NOT triggered.',
                '  S3 upload was NOT performed.',
            ]
        else:
            lines += [
                '  Post-clean validation FAILED.',
                '  Cleaned data S3 upload was NOT performed.',
            ]

        lines += ['', divider, '  Failed Checks:', '']
        for check, reason in details.get('failed_checks', {}).items():
            lines.append(f'    [{check}]')
            lines.append(f'      {reason}')
            lines.append('')

    elif stage == 'pipeline_crashed':
        lines += [
            '  An unexpected error caused the pipeline to crash.',
            '',
            divider,
            f"  Script  : {details.get('script', 'Unknown')}",
            f"  Error   : {details.get('error', 'Unknown')}",
            '',
            '  Traceback:',
            f"{details.get('traceback', 'No traceback available')}",
        ]

    lines += ['', divider, '— Scraper Diagnostics Bot']
    body = '\n'.join(lines)

    # ── Compose email ─────────────────────────────────────────────────────────
    msg            = EmailMessage()
    msg['From']    = sender_mail
    msg['To']      = receiver_mail
    msg['Subject'] = subject
    msg.set_content(body)

    # Attach log file if it exists
    if log_file_path and os.path.exists(log_file_path):
        with open(log_file_path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(log_file_path)
        msg.add_attachment(
            file_data,
            maintype='application',
            subtype='octet-stream',
            filename=file_name
        )

    # ── Send ──────────────────────────────────────────────────────────────────
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_mail, email_password)
    server.send_message(msg)
    server.quit()

    print(f'[notifications] Alert sent → {subject}')
