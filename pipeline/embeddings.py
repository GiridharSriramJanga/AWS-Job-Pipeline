#!/usr/bin/env python
# coding: utf-8

"""
embeddings.py
──────────────
Stage 4 of the pipeline — Job Description Embedding Generation.

Downloads the cleaned parquet from S3, generates chunked semantic
embeddings for each job description using all-MiniLM-L6-v2, and
uploads the result back to S3 under Embeddings_data/.

All cleaning, skill extraction, and feature engineering is handled
upstream in Jobsdata_cleaning_final_version.py — this script only
receives already-cleaned data and generates embeddings.

Chunking strategy:
    - Each job description is split into overlapping chunks
    - Each chunk carries a header: "{Job Title} at {Company}"
    - Chunks are batch encoded (128 at a time) for efficiency
    - Per-job chunk embeddings are stored as a list of lists

Batch processing note:
    batch_size=128 means 128 chunks are encoded as a single matrix
    operation — this is vectorized batching, not true parallelism.
    Sequential but significantly faster than one-by-one encoding.

Required environment variables:
    GROQ_API_KEY — not needed here, but set upstream
"""

import os
import sys
import re
import time
import traceback
import logging
import boto3
import pandas as pd
from datetime import datetime
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from notifications import send_alert

pd.set_option('display.max_columns', None)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────
download_dir = '/home/ec2-user/data_downloads'
date         = datetime.now().strftime('%m-%d-%Y')
bucket       = 'recentjobslistingbucket'
LOG_FILE     = os.path.join(download_dir, f'Job_scrape_{date}.log')

os.makedirs(download_dir, exist_ok=True)

# Embedding model settings
CONFIG = {
    'model_name'    : 'all-MiniLM-L6-v2',
    'max_tokens'    : 512,
    'chunk_size'    : 350,
    'overlap'       : 50,
    'header_tokens' : 50,
    'min_chunk_size': 20,
    'batch_size'    : 128,
}


# ── SemanticChunker ───────────────────────────────────────────────────────────
class SemanticChunker:
    """
    Splits job descriptions into overlapping token-aware chunks.
    Each chunk carries a header ("{title} at {company}") so the
    model has role context even when reading a partial description.
    """

    def __init__(self, model_name: str, chunk_size: int = 350, overlap: int = 50):
        self.tokenizer       = AutoTokenizer.from_pretrained(f'sentence-transformers/{model_name}')
        self.chunk_size      = chunk_size
        self.overlap         = overlap
        self.max_tokens      = CONFIG['max_tokens']
        self.header_tokens   = CONFIG['header_tokens']
        self.available_tokens = self.max_tokens - self.header_tokens

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def clean_text(self, text: str) -> str:
        """Minimal cleaning — strip HTML tags and normalise whitespace."""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        return text.strip()

    def split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def create_chunks_with_overlap(self, text: str, header: str = '') -> List[str]:
        """
        Build overlapping chunks from text.
        Each chunk prefixed with header to retain job context.
        """
        text      = self.clean_text(text)
        sentences = self.split_into_sentences(text)

        if not sentences:
            return []

        chunks          = []
        current_chunk   = []
        current_tokens  = 0
        header_token_count = self.count_tokens(header) if header else 0
        chunk_capacity  = min(self.chunk_size, self.available_tokens - header_token_count)

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # Sentence too long even for one chunk — split by raw tokens
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

            # Current chunk full — save it and start new one with overlap
            if current_tokens + sentence_tokens > chunk_capacity:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if header:
                        chunk_text = f'{header} {chunk_text}'
                    chunks.append(chunk_text)

                # Carry overlapping sentences into next chunk
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

        # Flush remaining
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if header:
                chunk_text = f'{header} {chunk_text}'
            chunks.append(chunk_text)

        return chunks

    def chunk_job_description(self, title: str, company: str, description: str) -> List[str]:
        """
        Entry point — chunk one job description.
        Returns a list of chunk strings, each prefixed with the job header.
        """
        header       = f'{title} at {company}.'
        desc_tokens  = self.count_tokens(description)
        header_tok   = self.count_tokens(header)
        total_tokens = desc_tokens + header_tok

        # Short enough to fit in one chunk — return directly
        if total_tokens <= (self.chunk_size + header_tok):
            return [f'{header} {self.clean_text(description)}']

        chunks = self.create_chunks_with_overlap(description, header)

        # Drop chunks below minimum useful size
        chunks = [c for c in chunks if self.count_tokens(c) >= CONFIG['min_chunk_size']]

        # Fallback — truncate safely if chunking produced nothing
        if not chunks:
            desc_clean       = self.clean_text(description)
            desc_token_list  = self.tokenizer.encode(desc_clean, add_special_tokens=False)
            max_content      = min(self.chunk_size, self.available_tokens - header_tok)
            safe_text        = self.tokenizer.decode(desc_token_list[:max_content])
            return [f'{header} {safe_text}']

        return chunks


# ── OptimizedChunkedEmbeddingGenerator ────────────────────────────────────────
class OptimizedChunkedEmbeddingGenerator:
    """
    Loads the sentence transformer model and generates batched
    chunk-level embeddings for a list of job descriptions.

    Batching note:
        Chunks from all jobs are pooled together and encoded 128
        at a time (batch_size). This is vectorized matrix batching,
        not parallel processing — sequential but efficient.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        logger.info(f'Loading embedding model: {model_name}')
        self.model              = SentenceTransformer(model_name)
        self.model.max_seq_length = CONFIG['max_tokens']
        self.chunker            = SemanticChunker(
            model_name=model_name,
            chunk_size=CONFIG['chunk_size'],
            overlap=CONFIG['overlap']
        )
        logger.info('Model loaded successfully')

    def generate_embeddings_batch(
        self,
        titles      : List[str],
        companies   : List[str],
        descriptions: List[str],
        batch_size  : int  = 128,
        show_progress: bool = True
    ) -> List[List[List[float]]]:
        """
        Generate chunked embeddings for all jobs.

        Steps:
            1. Chunk all job descriptions
            2. Encode all chunks in batches of batch_size
            3. Reassemble per-job chunk embedding lists

        Returns:
            List of length N (jobs), each element is a list of
            chunk embeddings (each embedding is 384-dim float list).
        """
        total_jobs = len(titles)
        logger.info(f'Starting embedding generation for {total_jobs} jobs')

        # ── Step 1: Chunk all jobs ────────────────────────────────────────────
        logger.info('Step 1/3: Chunking job descriptions...')

        all_chunks        = []
        chunk_to_job_map  = []
        chunks_per_job    = []

        for job_id, (title, company, desc) in enumerate(zip(titles, companies, descriptions)):
            try:
                title   = str(title)   if pd.notna(title)   else 'Unknown'
                company = str(company) if pd.notna(company) else 'Unknown'
                desc    = str(desc)    if pd.notna(desc)    else ''

                chunks = self.chunker.chunk_job_description(title, company, desc)

                all_chunks.extend(chunks)
                chunk_to_job_map.extend([job_id] * len(chunks))
                chunks_per_job.append(len(chunks))

            except Exception as e:
                logger.warning(f'Error chunking job {job_id}: {e}')
                chunks_per_job.append(0)

        total_chunks = len(all_chunks)
        avg_chunks   = total_chunks / total_jobs if total_jobs > 0 else 0
        logger.info(f'Created {total_chunks} chunks ({avg_chunks:.1f} avg per job)')

        # ── Step 2: Batch encode all chunks ───────────────────────────────────
        logger.info(f'Step 2/3: Encoding {total_chunks} chunks in batches of {batch_size}...')

        all_embeddings = self.model.encode(
            all_chunks,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress
        )

        logger.info(f'Generated {len(all_embeddings)} embeddings (dim: 384)')

        # ── Step 3: Reassemble per-job structure ──────────────────────────────
        logger.info('Step 3/3: Organising embeddings by job...')

        job_embeddings = []
        chunk_idx      = 0

        for num_chunks in chunks_per_job:
            if num_chunks == 0:
                job_embeddings.append([])
            else:
                job_chunk_embeddings = all_embeddings[chunk_idx:chunk_idx + num_chunks]
                job_embeddings.append(job_chunk_embeddings.tolist())
                chunk_idx += num_chunks

        storage_mb = (total_chunks * 384 * 4) / (1024 * 1024)
        logger.info(f'Embedding generation complete — estimated storage: {storage_mb:.2f} MB')

        return job_embeddings


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
            logger.info(f'Downloaded {cleaned_file_name} from S3')
        except Exception as e:
            raise RuntimeError(
                f'Failed to download cleaned file from S3 ({s3_key}). '
                f'Post-clean validation may not have completed. Error: {e}'
            )

        # ── Step 2: Load ──────────────────────────────────────────────────────────
        jobs_df = pd.read_parquet(local_path)
        logger.info(f'Loaded cleaned dataset: {jobs_df.shape}')

        # Ensure required columns are clean strings
        for col in ['Job Title', 'Company Name', 'Job Description']:
            if col not in jobs_df.columns:
                raise ValueError(f"Required column '{col}' not found in cleaned dataset.")
            jobs_df[col] = jobs_df[col].fillna('').astype(str).str.strip()

        # ── Step 3: Generate embeddings ───────────────────────────────────────────
        generator = OptimizedChunkedEmbeddingGenerator(model_name=CONFIG['model_name'])

        chunk_embeddings = generator.generate_embeddings_batch(
            titles       = jobs_df['Job Title'].tolist(),
            companies    = jobs_df['Company Name'].tolist(),
            descriptions = jobs_df['Job Description'].tolist(),
            batch_size   = CONFIG['batch_size'],
            show_progress=True
        )

        jobs_df['chunk_embeddings'] = chunk_embeddings
        logger.info(f'Embeddings added to dataframe — shape: {jobs_df.shape}')

        # ── Step 4: Validate embeddings before saving ──────────────────────────────
        jobs_missing_embeddings = sum(1 for emb in chunk_embeddings if len(emb) == 0)
        if jobs_missing_embeddings > 0:
            logger.warning(f'{jobs_missing_embeddings} jobs have empty embeddings')

        total_chunks = sum(len(emb) for emb in chunk_embeddings)
        avg_chunks   = total_chunks / len(jobs_df)
        emb_dim      = len(chunk_embeddings[0][0]) if chunk_embeddings and chunk_embeddings[0] else 'N/A'

        logger.info(f'Total chunks: {total_chunks}')
        logger.info(f'Average chunks per job: {avg_chunks:.2f}')
        logger.info(f'Embedding dimension: {emb_dim}')

        # ── Step 5: Save locally and upload to S3 ─────────────────────────────────
        embeddings_file_name = f'jobs_with_embeddings_{date}.parquet'
        embeddings_local     = os.path.join(download_dir, embeddings_file_name)

        jobs_df.to_parquet(embeddings_local, index=False)
        logger.info(f'Saved embeddings file to {embeddings_local}')

        try:
            s3.upload_file(embeddings_local, bucket, f'Embeddings_data/{embeddings_file_name}')
            logger.info(f'Uploaded {embeddings_file_name} to Embeddings_data/')
        except Exception as e:
            raise RuntimeError(f'Failed to upload embeddings file to S3: {e}')

        # ── Step 6: Cleanup local files ───────────────────────────────────────────
        for f in os.listdir(download_dir):
            fp = os.path.join(download_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)
                logger.info(f'Removed: {fp}')

        logger.info('Embeddings pipeline completed successfully.')

    except SystemExit:
        raise

    except Exception as e:
        send_alert(
            stage='pipeline_crashed',
            details={
                'script':    'embeddings.py',
                'error':     str(e),
                'traceback': traceback.format_exc()
            },
            log_file_path=LOG_FILE
        )
        sys.exit(1)
