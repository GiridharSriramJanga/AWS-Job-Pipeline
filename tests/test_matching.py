"""
test_matching.py
─────────────────
Unit tests for all scoring and extraction functions in
resume_matcher.py.

Test count:
    compute_semantic_score  4
    compute_skill_score     4
    extract_resume_skills   4
    hybrid score weights    1
    ──────────────────────────
    Total                  13
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

from pipeline.resume_matcher import (
    compute_semantic_score,
    compute_skill_score,
    extract_resume_skills,
    SEMANTIC_WEIGHT,
    SKILL_WEIGHT,
)


# ══════════════════════════════════════════════════════════════════════════════
# compute_semantic_score — 4 tests
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeSemanticScore:

    def test_identical_embeddings_return_score_near_one(
        self, mock_resume_embeddings
    ):
        """
        When resume and job embeddings are identical, cosine similarity
        should be ~1.0. Minor floating point variance is acceptable.
        """
        job_embeddings = mock_resume_embeddings.tolist()
        score          = compute_semantic_score(mock_resume_embeddings, job_embeddings)
        assert score >= 0.99

    def test_dissimilar_embeddings_return_low_score(
        self, mock_resume_embeddings, mock_job_embeddings_dissimilar
    ):
        """
        Randomly generated embeddings unrelated to the resume
        should produce a low cosine similarity score.
        """
        score = compute_semantic_score(
            mock_resume_embeddings,
            mock_job_embeddings_dissimilar
        )
        # Not necessarily 0 due to random vectors, but should be low
        assert score < 0.5

    def test_empty_job_embeddings_return_zero(self, mock_resume_embeddings):
        """Empty job embedding list returns 0.0 without crashing."""
        score = compute_semantic_score(mock_resume_embeddings, [])
        assert score == 0.0

    def test_similar_embeddings_return_high_score(
        self, mock_resume_embeddings, mock_job_embeddings_similar
    ):
        """
        Job embeddings that are slight perturbations of resume embeddings
        should produce a high similarity score.
        Max strategy rewards the best matching chunk — should be near 1.0.
        """
        score = compute_semantic_score(
            mock_resume_embeddings,
            mock_job_embeddings_similar
        )
        assert score >= 0.90


# ══════════════════════════════════════════════════════════════════════════════
# compute_skill_score — 4 tests
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeSkillScore:

    def _make_job_row(self, skills: dict) -> pd.Series:
        """
        Build a mock job row with binary skill columns.
        skills: {'tech_Python': 1, 'ops_AWS': 1, ...}
        """
        return pd.Series(skills)

    def test_perfect_skill_match_returns_one(self):
        """Resume has all skills the job requires — score should be 1.0."""
        resume_skills = {'python', 'sql', 'aws'}
        job_row       = self._make_job_row({
            'tech_Python': 1,
            'tech_SQL'   : 1,
            'ops_AWS'    : 1,
        })
        score = compute_skill_score(resume_skills, job_row)
        assert score == 1.0

    def test_zero_skill_overlap_returns_zero(self):
        """Resume has no skills the job requires — score should be 0.0."""
        resume_skills = {'tableau', 'excel'}
        job_row       = self._make_job_row({
            'tech_Python': 1,
            'tech_SQL'   : 1,
            'ops_AWS'    : 1,
        })
        score = compute_skill_score(resume_skills, job_row)
        assert score == 0.0

    def test_partial_skill_overlap_returns_proportion(self):
        """Resume has 2 of 4 required skills — score should be 0.5."""
        resume_skills = {'python', 'sql'}
        job_row       = self._make_job_row({
            'tech_Python'  : 1,
            'tech_SQL'     : 1,
            'ops_AWS'      : 1,
            'ops_Tableau'  : 1,
        })
        score = compute_skill_score(resume_skills, job_row)
        assert score == pytest.approx(0.5)

    def test_job_with_no_skills_returns_zero(self):
        """Job has no required skills (all binary columns 0) — returns 0.0."""
        resume_skills = {'python', 'sql'}
        job_row       = self._make_job_row({
            'tech_Python': 0,
            'tech_SQL'   : 0,
        })
        score = compute_skill_score(resume_skills, job_row)
        assert score == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# extract_resume_skills — 4 tests
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractResumeSkills:

    def test_common_skills_detected(self, sample_resume_text):
        """Python, SQL, Tableau, Pandas detected from standard resume text."""
        skills = extract_resume_skills(sample_resume_text)
        assert 'python' in skills
        assert 'sql'    in skills
        assert 'tableau' in skills
        assert 'pandas'  in skills

    def test_no_technical_skills_returns_empty_set(self, no_skills_resume_text):
        """Resume with no technical content returns an empty set."""
        skills = extract_resume_skills(no_skills_resume_text)
        assert isinstance(skills, set)
        assert len(skills) == 0

    def test_cloud_tools_detected(self):
        """AWS and GCP detected from resume mentioning cloud platforms."""
        resume = (
            "Worked extensively with AWS for data storage and processing. "
            "Experience with GCP BigQuery for large-scale analytics."
        )
        skills = extract_resume_skills(resume)
        assert 'aws'     in skills
        assert 'gcp'     in skills
        assert 'bigquery' in skills

    def test_empty_string_returns_empty_set(self, empty_resume_text):
        """Empty string input returns empty set without crashing."""
        skills = extract_resume_skills(empty_resume_text)
        assert isinstance(skills, set)
        assert len(skills) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Hybrid score weights — 1 test
# ══════════════════════════════════════════════════════════════════════════════

class TestHybridScoreWeights:

    def test_weights_sum_to_one(self):
        """
        SEMANTIC_WEIGHT + SKILL_WEIGHT must equal 1.0.
        If they don't, hybrid scores will be on the wrong scale.
        """
        assert SEMANTIC_WEIGHT + SKILL_WEIGHT == pytest.approx(1.0)
