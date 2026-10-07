#!/usr/bin/env python
# coding: utf-8

"""
resume_matcher.py
──────────────────
Matching engine called by the Streamlit app.

Designed as a class (ResumeMatcherEngine) so Streamlit can cache
the model loading with @st.cache_resource — models load once on
first call and stay in memory for all subsequent queries.

Full matching flow:
    1. Connect to S3, scan all parquet files in Embeddings_data/
    2. Apply metadata filters at S3 read time — never loads full dataset
    3. Chunk and encode resume using same model as JD embeddings
    4. Cosine similarity (max strategy) → semantic score per job
    5. Skill extraction from resume → overlap with job skill columns → skill score
    6. Hybrid score = (0.65 × semantic) + (0.35 × skill)
    7. Confidence banding:
           High   > 0.75 → return directly, sorted by hybrid score
           Medium 0.50–0.75 → cross-encoder reranked
           Low    < 0.50 → discarded
    8. Return ranked dataframe to Streamlit

CRITICAL — Model consistency:
    Resume and job descriptions MUST use the same embedding model.
    CONFIG['model_name'] here must always match CONFIG in embeddings.py.
    Cosine similarity is only valid within the same embedding space.

CRITICAL — Skill list consistency:
    Skill lists here must always match those in
    Jobsdata_cleaning_final_version.py. If you update skills there,
    update them here too.

    Both scripts have executable code at module level so direct import
    is not safe — kept in sync manually until a shared config module
    is created.
"""

import os
import re
import logging
import boto3
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Embedding config ──────────────────────────────────────────────────────────
# Must match CONFIG in embeddings.py exactly.
# If you change the model there, change it here too.
CONFIG = {
    'model_name'    : 'all-MiniLM-L6-v2',
    'max_tokens'    : 512,
    'chunk_size'    : 350,
    'overlap'       : 50,
    'header_tokens' : 50,
    'min_chunk_size': 20,
}

# Hybrid scoring weights — must sum to 1.0
SEMANTIC_WEIGHT  = 0.65
SKILL_WEIGHT     = 0.35

# Confidence thresholds
HIGH_THRESHOLD   = 0.75
MEDIUM_THRESHOLD = 0.50

# S3 config
BUCKET     = 'recentjobslistingbucket'
S3_PREFIX  = 'Embeddings_data/'
AWS_REGION = 'us-east-2'

# Cross-encoder — lightweight, pairs well with all-MiniLM-L6-v2
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# Date range options exposed to Streamlit
DATE_RANGES = {
    'Last day'     : 1,
    'Last week'    : 7,
    'Last month'   : 30,
    'Last 2 months': 60,
}


# ── Skill lists ───────────────────────────────────────────────────────────────
# Must stay in sync with Jobsdata_cleaning_final_version.py
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

ALL_SKILLS = technologies + operations + packages


# ── SemanticChunker ───────────────────────────────────────────────────────────
# Same class as in embeddings.py — must stay in sync.
# Resume must be chunked the same way as job descriptions so both
# live in the same embedding space.
class SemanticChunker:

    def __init__(self, model_name: str, chunk_size: int = 350, overlap: int = 50):
        self.tokenizer        = AutoTokenizer.from_pretrained(f'sentence-transformers/{model_name}')
        self.chunk_size       = chunk_size
        self.overlap          = overlap
        self.max_tokens       = CONFIG['max_tokens']
        self.header_tokens    = CONFIG['header_tokens']
        self.available_tokens = self.max_tokens - self.header_tokens

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def clean_text(self, text: str) -> str:
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        return text.strip()

    def split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def create_chunks_with_overlap(self, text: str, header: str = '') -> List[str]:
        text      = self.clean_text(text)
        sentences = self.split_into_sentences(text)

        if not sentences:
            return []

        chunks         = []
        current_chunk  = []
        current_tokens = 0
        header_tok     = self.count_tokens(header) if header else 0
        chunk_capacity = min(self.chunk_size, self.available_tokens - header_tok)

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if sentence_tokens > chunk_capacity:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if header:
                        chunk_text = f'{header} {chunk_text}'
                    chunks.append(chunk_text)
                    current_chunk  = []
                    current_tokens = 0

                tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
                for j in range(0, len(tokens), chunk_capacity):
                    chunk_tokens = tokens[j:j + chunk_capacity]
                    chunk_text   = self.tokenizer.decode(chunk_tokens)
                    if header:
                        chunk_text = f'{header} {chunk_text}'
                    chunks.append(chunk_text)
                continue

            if current_tokens + sentence_tokens > chunk_capacity:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if header:
                        chunk_text = f'{header} {chunk_text}'
                    chunks.append(chunk_text)

                overlap_sentences = []
                overlap_tokens    = 0
                for sent in reversed(current_chunk):
                    sent_tokens = self.count_tokens(sent)
                    if overlap_tokens + sent_tokens <= self.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break

                current_chunk  = overlap_sentences
                current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if header:
                chunk_text = f'{header} {chunk_text}'
            chunks.append(chunk_text)

        return chunks

    def chunk_resume(self, resume_text: str) -> List[str]:
        """
        Chunk a resume text.
        No header prefix — resume speaks for itself unlike JD chunks
        which need job title context.
        """
        text_clean   = self.clean_text(resume_text)
        total_tokens = self.count_tokens(text_clean)

        # Short enough for one chunk
        if total_tokens <= self.chunk_size:
            return [text_clean]

        chunks = self.create_chunks_with_overlap(resume_text)
        chunks = [c for c in chunks if self.count_tokens(c) >= CONFIG['min_chunk_size']]

        # Fallback — truncate safely
        if not chunks:
            tokens    = self.tokenizer.encode(text_clean, add_special_tokens=False)
            safe_text = self.tokenizer.decode(tokens[:self.chunk_size])
            return [safe_text]

        return chunks


# ── Scoring functions ─────────────────────────────────────────────────────────
def compute_semantic_score(
    resume_embeddings: np.ndarray,
    job_embeddings   : list
) -> float:
    """
    Max similarity strategy:
        For each resume chunk, find the highest cosine similarity
        across all job chunks. Average these max scores.

    Why max not mean:
        A resume's Python section should strongly match the JD's
        requirements section — not be diluted by the JD's company
        culture or benefits sections which are irrelevant.

    Embeddings are already L2-normalised (normalize_embeddings=True
    in embeddings.py) so dot product == cosine similarity.
    """
    if not job_embeddings:
        return 0.0

    job_emb_array = np.array(job_embeddings)      # (n_job_chunks, 384)

    # (n_resume_chunks, 384) × (384, n_job_chunks) → (n_resume_chunks, n_job_chunks)
    similarity_matrix = np.dot(resume_embeddings, job_emb_array.T)

    # Best matching job chunk for each resume chunk
    max_per_resume_chunk = similarity_matrix.max(axis=1)  # (n_resume_chunks,)

    return float(max_per_resume_chunk.mean())


def extract_resume_skills(resume_text: str) -> set:
    """
    Extract canonical skill names from resume text.
    Uses same skill lists as cleaning script — simple word boundary matching.
    Returns a set of lowercase canonical skill names.
    """
    if not isinstance(resume_text, str):
        return set()

    resume_lower = resume_text.lower()
    found        = set()

    for skill in ALL_SKILLS:
        if re.search(rf'\b{re.escape(skill.lower())}\b', resume_lower):
            found.add(skill.lower())

    return found


def compute_skill_score(resume_skills: set, job_row: pd.Series) -> float:
    """
    Skill match score: what proportion of the job's required skills
    does the resume cover.

    Answers: "how qualified is this person for this specific job?"

    Uses binary skill columns (tech_*, ops_*, lib_*) from cleaned dataset.
    Ignores the 'None' placeholder columns.
    """
    skill_cols = [
        col for col in job_row.index
        if (col.startswith('tech_') or col.startswith('ops_') or col.startswith('lib_'))
        and not col.endswith('_None')
    ]

    # Get skills this job actually requires
    job_required_skills = set()
    for col in skill_cols:
        if job_row[col] == 1:
            # e.g. tech_Python → python, ops_AWS → aws
            skill_name = col.split('_', 1)[1].lower()
            job_required_skills.add(skill_name)

    if not job_required_skills:
        return 0.0

    matched = resume_skills.intersection(job_required_skills)
    return len(matched) / len(job_required_skills)


def rerank_with_cross_encoder(
    medium_df     : pd.DataFrame,
    resume_text   : str,
    cross_encoder : CrossEncoder
) -> pd.DataFrame:
    """
    Rerank medium confidence jobs using a cross-encoder.

    Cross-encoder encodes resume + JD together rather than separately,
    giving a more accurate relevance score at the cost of speed.
    Only called on medium confidence jobs — not the full dataset.

    Scores are min-max normalised to 0-1 for interpretability.
    """
    resume_truncated = resume_text[:1000]

    pairs = [
        (resume_truncated, str(row.get('Job Description', ''))[:1500])
        for _, row in medium_df.iterrows()
    ]

    ce_scores = cross_encoder.predict(pairs)

    medium_df                     = medium_df.copy()
    medium_df['cross_encoder_score'] = ce_scores

    # Normalise to 0-1 range
    if len(ce_scores) > 1:
        min_s = ce_scores.min()
        max_s = ce_scores.max()
        if max_s > min_s:
            medium_df['cross_encoder_score'] = (
                (ce_scores - min_s) / (max_s - min_s)
            )

    return medium_df.sort_values('cross_encoder_score', ascending=False)


# ── S3 data loader ────────────────────────────────────────────────────────────
def load_and_filter_from_s3(
    s3_client : boto3.client,
    role      : str,
    date_range: str,
    experience: Optional[int]
) -> pd.DataFrame:
    """
    Scan all parquet files in S3 Embeddings_data/ and return only
    rows matching the user's filters. Filters applied at read time
    so the full dataset is never loaded into memory.

    Deduplicates across files — same job may exist in multiple
    scrape date files.

    Parameters
    ----------
    role       : job role string matching Job_Title_Cleaned values
    date_range : one of the DATE_RANGES keys
    experience : user's years of experience (used as upper bound filter)
    """
    response = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=S3_PREFIX)

    if 'Contents' not in response:
        logger.warning('No files found in S3 Embeddings_data/')
        return pd.DataFrame()

    days_back   = DATE_RANGES.get(date_range, 7)
    cutoff_date = datetime.now() - timedelta(days=days_back)

    filtered_dfs = []

    for obj in response['Contents']:
        key = obj['Key']

        if not key.endswith('.parquet'):
            continue

        try:
            buffer = BytesIO()
            s3_client.download_fileobj(BUCKET, key, buffer)
            buffer.seek(0)
            df = pd.read_parquet(buffer)

            # ── Date filter ───────────────────────────────────────────────────
            if 'Scrapped_date' in df.columns:
                df['Scrapped_date'] = pd.to_datetime(df['Scrapped_date'], errors='coerce')
                df = df[df['Scrapped_date'] >= cutoff_date]

            # ── Role filter ───────────────────────────────────────────────────
            if role and 'Job_Title_Cleaned' in df.columns:
                df = df[df['Job_Title_Cleaned'].str.lower() == role.lower()]

            # ── Experience filter ─────────────────────────────────────────────
            # Include jobs where:
            #   - required experience <= user experience + 1 (slight tolerance)
            #   - OR experience is unknown (None) — don't discard unknowns
            if experience is not None and 'Experience' in df.columns:
                df = df[
                    df['Experience'].isna() |
                    (df['Experience'] <= experience + 1)
                ]

            if len(df) > 0:
                filtered_dfs.append(df)
                logger.info(f'[s3] {len(df)} matching jobs from {key}')

        except Exception as e:
            logger.warning(f'[s3] Failed to read {key}: {e}')
            continue

    if not filtered_dfs:
        logger.info('[s3] No jobs matched filters across all files')
        return pd.DataFrame()

    combined = pd.concat(filtered_dfs, ignore_index=True)

    # Deduplicate across scrape dates — same job posted across multiple scrapes
    before_dedup = len(combined)
    combined     = combined.drop_duplicates(
        subset=['Company Name', 'Job Title', 'Location']
    )
    logger.info(
        f'[s3] {len(combined)} jobs after dedup '
        f'(removed {before_dedup - len(combined)} cross-file duplicates)'
    )

    return combined.reset_index(drop=True)


# ── Matcher engine ────────────────────────────────────────────────────────────
class ResumeMatcherEngine:
    """
    Main matching engine. Instantiate once and cache in Streamlit:

        @st.cache_resource
        def get_matcher():
            return ResumeMatcherEngine()

    Both models (embedding + cross-encoder) load on first instantiation
    and stay in memory for the duration of the Streamlit session.
    Cold start: ~8-12 seconds. Subsequent queries: ~1-5 seconds.
    """

    def __init__(self):
        logger.info('[engine] Loading embedding model...')
        self.embedding_model              = SentenceTransformer(CONFIG['model_name'])
        self.embedding_model.max_seq_length = CONFIG['max_tokens']

        logger.info('[engine] Loading cross-encoder model...')
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

        self.chunker = SemanticChunker(
            model_name=CONFIG['model_name'],
            chunk_size=CONFIG['chunk_size'],
            overlap   =CONFIG['overlap']
        )

        self.s3 = boto3.client('s3', region_name=AWS_REGION)

        logger.info('[engine] ResumeMatcherEngine ready')

    def match(
        self,
        resume_text: str,
        role       : str,
        date_range : str,
        experience : Optional[int] = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Main matching function called by Streamlit.

        Parameters
        ----------
        resume_text : plain text extracted from user's uploaded resume
        role        : e.g. 'Data Analyst', 'Data Scientist'
        date_range  : one of 'Last day', 'Last week', 'Last month', 'Last 2 months'
        experience  : user's years of experience (int) or None

        Returns
        -------
        ranked_df   : matched and ranked jobs dataframe
        summary     : dict with match statistics for display in Streamlit
        """

        # ── Step 1: Load and filter from S3 ──────────────────────────────────
        logger.info(f'[match] Filters — role: {role} | date: {date_range} | exp: {experience}')
        jobs_df = load_and_filter_from_s3(self.s3, role, date_range, experience)

        if jobs_df.empty:
            logger.info('[match] No jobs found for given filters')
            return pd.DataFrame(), {'status': 'no_jobs', 'total_filtered': 0}

        logger.info(f'[match] {len(jobs_df)} jobs loaded after filtering')

        # ── Step 2: Chunk and encode resume ───────────────────────────────────
        resume_chunks = self.chunker.chunk_resume(resume_text)

        if not resume_chunks:
            logger.warning('[match] Resume produced no chunks')
            return pd.DataFrame(), {'status': 'resume_error'}

        resume_embeddings = self.embedding_model.encode(
            resume_chunks,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        logger.info(f'[match] Resume encoded → {len(resume_chunks)} chunks')

        # ── Step 3: Extract skills from resume ────────────────────────────────
        resume_skills = extract_resume_skills(resume_text)
        logger.info(f'[match] Resume skills found: {resume_skills}')

        # ── Step 4: Score every job ───────────────────────────────────────────
        semantic_scores = []
        skill_scores    = []

        for _, job_row in jobs_df.iterrows():
            job_embeddings = job_row.get('chunk_embeddings', [])
            semantic_scores.append(
                compute_semantic_score(resume_embeddings, job_embeddings)
            )
            skill_scores.append(
                compute_skill_score(resume_skills, job_row)
            )

        jobs_df                 = jobs_df.copy()
        jobs_df['semantic_score'] = semantic_scores
        jobs_df['skill_score']    = skill_scores
        jobs_df['hybrid_score']   = (
            SEMANTIC_WEIGHT * jobs_df['semantic_score'] +
            SKILL_WEIGHT    * jobs_df['skill_score']
        )

        # ── Step 5: Confidence banding ────────────────────────────────────────
        def band(score):
            if score >= HIGH_THRESHOLD:
                return 'high'
            elif score >= MEDIUM_THRESHOLD:
                return 'medium'
            else:
                return 'low'

        jobs_df['confidence_band'] = jobs_df['hybrid_score'].apply(band)

        total_before_filter = len(jobs_df)

        # Discard low confidence
        jobs_df = jobs_df[jobs_df['confidence_band'] != 'low'].copy()

        logger.info(
            f'[match] Confidence bands — '
            f"high: {(jobs_df['confidence_band'] == 'high').sum()} | "
            f"medium: {(jobs_df['confidence_band'] == 'medium').sum()} | "
            f"low discarded: {total_before_filter - len(jobs_df)}"
        )

        if jobs_df.empty:
            logger.info('[match] No jobs passed confidence threshold')
            return pd.DataFrame(), {
                'status'         : 'no_matches',
                'total_filtered' : total_before_filter,
                'resume_skills'  : list(resume_skills)
            }

        # ── Step 6: Split by confidence and rerank medium ─────────────────────
        high_df   = jobs_df[jobs_df['confidence_band'] == 'high'].copy()
        medium_df = jobs_df[jobs_df['confidence_band'] == 'medium'].copy()

        # Sort high confidence by hybrid score
        high_df = high_df.sort_values('hybrid_score', ascending=False)

        # Rerank medium confidence with cross-encoder
        if not medium_df.empty:
            logger.info(f'[match] Reranking {len(medium_df)} medium confidence jobs')
            medium_df = rerank_with_cross_encoder(medium_df, resume_text, self.cross_encoder)

        # ── Step 7: Combine — high first, then medium ─────────────────────────
        final_df = pd.concat([high_df, medium_df], ignore_index=True)

        # ── Step 8: Select output columns ─────────────────────────────────────
        output_cols = [
            'Job Title',
            'Company Name',
            'Location',
            'Salary Estimate',
            'Job_Title_Cleaned',
            'Seniority_simplified',
            'Experience',
            'Scrapped_date',
            'hybrid_score',
            'semantic_score',
            'skill_score',
            'confidence_band',
        ]

        if 'cross_encoder_score' in final_df.columns:
            output_cols.append('cross_encoder_score')

        available_cols = [col for col in output_cols if col in final_df.columns]
        final_df       = final_df[available_cols].reset_index(drop=True)

        # Round scores for display
        score_cols = ['hybrid_score', 'semantic_score', 'skill_score', 'cross_encoder_score']
        for col in score_cols:
            if col in final_df.columns:
                final_df[col] = final_df[col].round(4)

        summary = {
            'status'          : 'success',
            'total_filtered'  : total_before_filter,
            'total_matched'   : len(final_df),
            'high_confidence' : int((final_df['confidence_band'] == 'high').sum()),
            'medium_confidence': int((final_df['confidence_band'] == 'medium').sum()),
            'resume_skills'   : sorted(list(resume_skills)),
            'resume_chunks'   : len(resume_chunks),
        }

        logger.info(f"[match] Done — {len(final_df)} jobs returned")
        return final_df, summary
