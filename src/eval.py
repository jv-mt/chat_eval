"""
LLM Evaluation Framework

This module provides a comprehensive framework for evaluating Large Language Models (LLMs)
using RAGAS metrics with Ollama integration. It enables automated assessment of RAG
(Retrieval-Augmented Generation) systems by measuring answer quality, context relevance,
and faithfulness to source material.

Key Features:
- Evaluates LLM responses against ground truth using multiple RAGAS metrics
- Integrates with Ollama for local LLM inference and embedding generation
- Processes CSV files containing model outputs and reference answers
- Supports multiple judge models for comparative evaluation
- Provides detailed performance metrics and timing statistics
- Outputs structured JSON results for further analysis

Usage:
    python eval.py [--results RESULTS_CSV] [--ground-truth GROUND_TRUTH_CSV] [--output OUTPUT_FILE]

Author: Jukka Veijanen
"""

from datetime import datetime
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from datasets import Dataset

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    AnswerAccuracy,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextRelevance,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference,
)
from ragas.exceptions import RagasOutputParserException

# from config import get, get_logger, setup_logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get, get_logger, setup_logging

# Load environment variables from .env file
load_dotenv()


class EvaluationConfig:
    """Configuration class for evaluation settings."""

    def __init__(self):
        """Initialize configuration from settings and environment variables."""
        # Ollama configuration
        self.ollama_host = os.getenv("OLLAMA_HOST") or get(
            "ollama.host", "http://localhost:11434"
        )
        self.embedding_model = os.getenv("EMBEDDING_MODEL") or get(
            "models.embedding.default", "nomic-embed-text"
        )

        # Judge models
        judge_model_env = os.getenv("JUDGE_MODEL_LIST")
        if judge_model_env:
            self.judge_models = judge_model_env.split(",")
        else:
            self.judge_models = get(
                "models.judge.available",
            )

        # File paths
        self.results_csv = get("paths.inputs.results")
        self.ground_truth_csv = get("paths.inputs.ground_truth")
        self.output_file = get("paths.outputs.evaluation_output")

        # Evaluation settings
        self.temperature = get("evaluation.temperature", 0.0)
        self.timeout = get("evaluation.timeout", 300)
        self.max_retries = get("evaluation.max_retries", 3)


def get_evaluation_metrics() -> List[Any]:
    """
    Get the list of RAGAS metrics to use for evaluation.

    Returns:
        List[Any]: List of RAGAS metric classes from configuration or defaults.
    """
    # Get metrics from config or use defaults
    metric_names = get(
        "evaluation.metrics",
        [
            "AnswerAccuracy",
            "AnswerRelevancy",
            "ContextPrecision",
            "ContextRecall",
            "ContextRelevance",
            "Faithfulness",
            "LLMContextPrecisionWithoutReference",
            "LLMContextPrecisionWithReference",
        ],
    )

    # Map metric names to classes
    metric_map = {
        "AnswerAccuracy": AnswerAccuracy,
        "AnswerRelevancy": AnswerRelevancy,
        "ContextPrecision": ContextPrecision,
        "ContextRecall": ContextRecall,
        "ContextRelevance": ContextRelevance,
        "Faithfulness": Faithfulness,
        "LLMContextPrecisionWithoutReference": LLMContextPrecisionWithoutReference,
        "LLMContextPrecisionWithReference": LLMContextPrecisionWithReference,
    }

    return [metric_map[name] for name in metric_names if name in metric_map]


def log_configuration(logger, config: EvaluationConfig) -> None:
    """
    Log the current configuration settings for the evaluation system.

    Args:
        logger: Logger instance to use for output.
        config: Configuration instance containing settings.
    """
    logger.info("Starting evaluation with configuration:")
    logger.info("  OLLAMA_HOST: %s", config.ollama_host)
    logger.info("  JUDGE_MODELS: %s", config.judge_models)
    logger.info("  EMBEDDING_MODEL: %s", config.embedding_model)
    logger.info("  Total judge models: %d", len(config.judge_models))
    logger.info("  Results CSV: %s", config.results_csv)
    logger.info("  Ground Truth CSV: %s", config.ground_truth_csv)
    logger.info("  Output File: %s", config.output_file)


def parse_document_contexts(documents: pd.Series) -> List[List[str]]:
    """
    Parse document strings into context lists for RAGAS evaluation.

    This function processes document strings that are expected to be in a specific
    format with quoted strings separated by commas. It extracts the first two
    documents from each entry to create context lists.

    Args:
        documents (pd.Series): Pandas Series containing document strings in format:
                              '["doc1", "doc2", ...]' or similar quoted format.

    Returns:
        List[List[str]]: List of context lists, where each inner list contains
                        up to 2 document strings. Returns ["", ""] for malformed entries.

    Note:
        The current parsing logic assumes a specific string format and may fail
        on differently formatted document strings. Consider using json.loads()
        for more robust parsing if documents are JSON-formatted.
    """
    logger = get_logger(__name__)
    logger.debug("Processing %d document strings", len(documents))

    retrieved_contexts = []

    for idx, document in enumerate(documents):
        try:
            contexts = document
            docs = []
            docs.append(contexts[1:].split('", "')[0][1:])
            docs.append(contexts[:-2].split('", "')[1][:-2])
            retrieved_contexts.append(docs)
        except Exception as e:
            logger.error("Error processing document at index %d: %s", idx, e)
            logger.debug("Document content: %s", document)
            retrieved_contexts.append(["", ""])

    logger.debug("Successfully processed %d document contexts", len(retrieved_contexts))
    return retrieved_contexts


def load_and_merge_data(config: EvaluationConfig) -> pd.DataFrame:
    """
    Load and merge evaluation data from CSV files.

    Loads results CSV and ground truth CSV files, then merges them on the
    query/Question columns. The merge is performed as a left join to preserve
    all results entries.

    Args:
        config (EvaluationConfig): Configuration instance containing file paths.

    Returns:
        pd.DataFrame: Merged DataFrame containing columns from both files:
                     - query: Input questions/prompts
                     - response: Model generated answers
                     - model: Name of the model used
                     - documents: Retrieved context documents
                     - Ground Truth: Reference/correct answers
                     Note: The redundant 'Question' column is removed after merge.

    Raises:
        FileNotFoundError: If either CSV file cannot be found.
        pd.errors.EmptyDataError: If CSV files are empty.
        Exception: For other file reading or merging errors.
    """
    logger = get_logger(__name__)
    logger.info("Loading data files:")
    logger.info("  Results: %s", config.results_csv)
    logger.info("  Ground Truth: %s", config.ground_truth_csv)

    try:
        # Load results CSV
        results_df = pd.read_csv(config.results_csv)
        logger.info("Loaded %d rows from results file", len(results_df))

        # Load ground truth CSV
        ground_truth_df = pd.read_csv(config.ground_truth_csv)
        logger.info("Loaded %d rows from ground truth file", len(ground_truth_df))

        # Merge on query/Question columns
        merged_df = results_df.merge(
            ground_truth_df, left_on="query", right_on="Question", how="left"
        )

        # Remove redundant Question column
        merged_df = merged_df.drop(columns=["Question"])

        logger.info("Successfully merged data: %d rows", len(merged_df))
        return merged_df

    except Exception as e:
        logger.error("Error loading and merging data: %s", e)
        raise


def prepare_ragas_dataset(df: pd.DataFrame) -> Dataset:
    """
    Convert merged DataFrame to RAGAS-compatible dataset format.

    Transforms the merged evaluation data into a Dataset with
    the structure required by RAGAS metrics. Parses document contexts and
    maps DataFrame columns to RAGAS expected field names.

    Args:
        df (pd.DataFrame): Merged DataFrame with evaluation data containing:
                          - query: Input questions
                          - response: Model answers
                          - Ground Truth: Reference answers
                          - model: Model names
                          - documents: Context documents (to be parsed)

    Returns:
        Dataset: Dataset with fields:
                - question: Input questions
                - answer: Model responses
                - ground_truth: Reference answers
                - model: Model names
                - contexts: Parsed document contexts as List[List[str]]
    """
    logger = get_logger(__name__)
    logger.info("Preparing RAGAS dataset from %d samples", len(df))

    # Parse document contexts
    retrieved_contexts = parse_document_contexts(df["documents"])

    # Prepare data dictionary for RAGAS
    data = {
        "question": df["query"].tolist(),
        "answer": df["response"].tolist(),
        "ground_truth": df["Ground Truth"].tolist(),
        "model": df["model"].tolist(),
        "contexts": retrieved_contexts,
    }

    dataset = Dataset.from_dict(data)
    logger.info("Created RAGAS dataset with %d samples", len(dataset))
    return dataset


def initialize_models(
    config: EvaluationConfig, judge_model: str
) -> Tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """
    Initialize the judge LLM and embedding models for evaluation.

    Args:
        config: Configuration instance.
        judge_model: Name of the judge model to initialize.

    Returns:
        Tuple containing wrapped LLM and embeddings models.
    """
    logger = get_logger(__name__)
    logger.info("Initializing models for judge: %s", judge_model)

    try:
        llm = ChatOllama(
            base_url=config.ollama_host,
            format="json",
            model=judge_model,
            temperature=config.temperature,
        )
        evaluator_llm = LangchainLLMWrapper(llm)
        logger.info("Successfully initialized judge model: %s", judge_model)
    except Exception as e:
        logger.error("Failed to initialize judge model %s: %s", judge_model, e)
        raise

    try:
        embeddings = OllamaEmbeddings(
            model=config.embedding_model, base_url=config.ollama_host
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
        logger.info(
            "Successfully initialized embedding model: %s", config.embedding_model
        )
    except Exception as e:
        logger.error("Failed to initialize embedding model: %s", e)
        raise

    return evaluator_llm, evaluator_embeddings


def evaluate_single_sample(
    sample: SingleTurnSample,
    metrics: List[Any],
    evaluator_llm: LangchainLLMWrapper,
    evaluator_embeddings: LangchainEmbeddingsWrapper,
    sample_index: int,
    model_name: str,
) -> List[Dict[str, Any]]:
    """
    Evaluate a single sample against all specified metrics.

    Applies each RAGAS metric to the given sample, handling both LLM-only
    and LLM+embeddings metrics. Gracefully handles evaluation failures
    by logging errors and continuing with remaining metrics.

    Args:
        sample (SingleTurnSample): RAGAS sample object containing:
                                  - user_input: Question/prompt
                                  - response: Model answer
                                  - reference: Ground truth answer
                                  - retrieved_contexts: Context documents
        metrics (List[Any]): List of RAGAS metric classes to evaluate.
        evaluator_llm (LangchainLLMWrapper): Wrapped LLM for evaluation.
        evaluator_embeddings (LangchainEmbeddingsWrapper): Wrapped embeddings model.
        sample_index (int): Index of the sample (for logging).
        model_name (str): Name of the model that generated the response.

    Returns:
        List[Dict[str, Any]]: List of metric results, where each dict contains:
                             - metric: Name of the metric class
                             - score: Numeric evaluation score

    Note:
        Failed metric evaluations are logged but don't stop the process.
        RagasOutputParserException and other errors are handled gracefully.
    """
    logger = get_logger(__name__)
    logger.debug("Evaluating sample %d for model %s", sample_index, model_name)

    results = []

    for metric_class in metrics:
        try:
            # Initialize metric with both LLM and embeddings
            try:
                metric = metric_class(
                    llm=evaluator_llm, embeddings=evaluator_embeddings
                )
            except TypeError:
                # Fallback to LLM only if embeddings not supported
                metric = metric_class(llm=evaluator_llm)

            # Evaluate the sample
            result = metric.single_turn_score(sample)

            results.append({"metric": metric_class.__name__, "score": result})
            logger.info(f"Result: metric: {metric_class.__name__}, score: {result}")

        except RagasOutputParserException as e:
            logger.warning(
                "Parser error for metric %s on sample %d: %s",
                metric_class.__name__,
                sample_index,
                e,
            )
        except Exception as e:
            logger.error(
                "Error evaluating metric %s on sample %d: %s",
                metric_class.__name__,
                sample_index,
                e,
            )

    return results


def evaluate_dataset_with_judge(
    eval_dataset: Dataset, config: EvaluationConfig, judge_model: str
) -> List[Dict[str, Any]]:
    """
    Evaluate entire dataset using a specific judge model.

    Orchestrates the complete evaluation process for all samples in the dataset
    using a single judge model. Initializes models, processes each sample through
    all configured metrics, and tracks timing statistics.

    Args:
        eval_dataset (Dataset): Dataset containing samples with fields:
                               - question: Input questions
                               - answer: Model responses
                               - ground_truth: Reference answers
                               - model: Model names
                               - contexts: Context documents
        config (EvaluationConfig): Configuration instance with model settings.
        judge_model (str): Name of the judge model for evaluation.

    Returns:
        List[Dict[str, Any]]: Evaluation results for all samples, where each dict contains:
                             - model: Name of the model that generated the response
                             - question: Input question/prompt
                             - metrics: List of metric evaluation results

    Raises:
        Exception: If model initialization fails or critical errors occur.
                  Individual sample/metric failures are handled gracefully.
    """
    logger = get_logger(__name__)
    logger.info("Starting evaluation with judge model: %s", judge_model)

    start_time = time.time()

    # Initialize models
    evaluator_llm, evaluator_embeddings = initialize_models(config, judge_model)

    # Get metrics to evaluate
    metrics = get_evaluation_metrics()
    logger.info("Using %d metrics: %s", len(metrics), [m.__name__ for m in metrics])

    results = []
    total_samples = len(eval_dataset)

    for idx, sample_data in enumerate(eval_dataset):
        logger.info("Processing sample %d/%d", idx + 1, total_samples)

        # Create RAGAS sample
        sample = SingleTurnSample(
            user_input=sample_data["question"],
            response=sample_data["answer"],
            reference=sample_data["ground_truth"],
            retrieved_contexts=sample_data["contexts"],
        )

        # Evaluate sample
        sample_results = evaluate_single_sample(
            sample,
            metrics,
            evaluator_llm,
            evaluator_embeddings,
            idx,
            sample_data["model"],
        )

        results.append(
            {
                "model": sample_data["model"],
                "question": sample_data["question"],
                "metrics": sample_results,
            }
        )
        logger.info(f"Sample results: {sample_results}")

    elapsed_time = time.time() - start_time
    logger.info(
        "Completed evaluation with %s in %.2f seconds", judge_model, elapsed_time
    )

    return results


def save_evaluation_results(
    results: List[Dict[str, Any]], config: EvaluationConfig
) -> None:
    """
    Save evaluation results to JSON file with incremental updates.

    Loads existing evaluation results if the output file exists, appends
    new results, and saves the combined data. This allows for incremental
    evaluation runs without losing previous results.

    Args:
        results (List[Dict[str, Any]]): New evaluation results to save.
                                       Each result should contain:
                                       - judge_model: Name of judge model
                                       - timestamp: Evaluation timestamp
                                       - total_samples: Number of samples
                                       - results: List of sample evaluations
        config (EvaluationConfig): Configuration instance with output file path.

    Raises:
        json.JSONDecodeError: If existing file contains invalid JSON (handled gracefully).
        IOError: If file operations fail due to permissions or disk space.
        Exception: For other unexpected errors during save operation.

    Note:
        Creates output directory if it doesn't exist. Uses UTF-8 encoding
        and pretty-printing for human readability.
    """
    logger = get_logger(__name__)
    logger.info("Saving evaluation results to %s", config.output_file)

    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(config.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            logger.debug("Created output directory: %s", output_dir)

        # Load existing results if file exists
        existing_results = []
        if os.path.exists(config.output_file):
            try:
                with open(config.output_file, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                logger.info("Loaded %d existing evaluation runs", len(existing_results))
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("Could not load existing results: %s", e)
                existing_results = []

        # Append new results
        existing_results.extend(results)

        # Save updated results
        with open(config.output_file, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, indent=2, ensure_ascii=False)

        logger.info(
            "Successfully saved %d evaluation runs to %s",
            len(existing_results),
            config.output_file,
        )

    except Exception as e:
        logger.error("Failed to save final results: %s", e)
        raise


def run_evaluation_pipeline(config: EvaluationConfig) -> None:
    """
    Run the complete evaluation pipeline from data loading to results saving.

    Orchestrates the entire evaluation process: loads and merges input data,
    prepares RAGAS dataset, evaluates with each configured judge model,
    and saves results incrementally. Handles failures gracefully by continuing
    with remaining judge models.

    Args:
        config (EvaluationConfig): Configuration instance containing:
                                  - File paths for input/output
                                  - Judge models list
                                  - Evaluation settings

    Raises:
        Exception: Critical errors that prevent pipeline execution are re-raised.
                  Individual judge model failures are logged but don't stop the pipeline.

    Note:
        - Saves evaluation dataset to 'eval_dataset.csv' for debugging
        - Results are saved incrementally after each judge model
        - Pipeline timing is logged for performance monitoring
    """
    logger = get_logger(__name__)

    # Ensure logs directory exists for file logging
    os.makedirs("logs", exist_ok=True)
    logger.debug("Created logs directory if it didn't exist")

    # Test logging at all levels to verify configuration
    logger.debug(
        "DEBUG: Starting evaluation pipeline - this should appear in files only"
    )
    logger.info(
        "INFO: Pipeline initialization - this should appear in console and files"
    )
    logger.warning(
        "WARNING: Test warning message - this should appear in console and files"
    )

    start_time = time.time()
    logger.info(
        "Starting evaluation pipeline at %s", time.strftime("%Y-%m-%d %H:%M:%S")
    )

    pipeline_start_time = time.time()

    try:
        # Load and merge data
        merged_df = load_and_merge_data(config)

        # Prepare RAGAS dataset
        eval_dataset = prepare_ragas_dataset(merged_df)

        # Save dataset for debugging
        eval_dataset.to_csv("eval_dataset.csv")
        logger.info("Saved evaluation dataset to eval_dataset.csv for debugging")

        # Evaluate with each judge model
        for judge_model in config.judge_models:
            logger.info("Evaluating with judge model: %s", judge_model)

            try:
                # Run evaluation
                judge_results = evaluate_dataset_with_judge(
                    eval_dataset, config, judge_model
                )

                # Add metadata
                evaluation_record = {
                    "judge_model": judge_model,
                    "timestamp": datetime.now().isoformat(),
                    "total_samples": len(judge_results),
                    "results": judge_results,
                }

                logger.info("Completed evaluation with judge: %s", judge_model)
                save_evaluation_results([evaluation_record], config)
                logger.info("Saved results for judge: %s", judge_model)

            except Exception as e:
                logger.error("Failed evaluation with judge %s: %s", judge_model, e)
                continue

        pipeline_elapsed = time.time() - pipeline_start_time
        logger.info("Pipeline completed in %.2f seconds", pipeline_elapsed)

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        raise


def main() -> None:
    """Main execution function."""
    # Setup logging
    logger = setup_logging()

    # Initialize configuration
    config = EvaluationConfig()

    # Log configuration
    log_configuration(logger, config)

    try:
        # Run the evaluation pipeline
        run_evaluation_pipeline(config)
        logger.info("Evaluation process completed successfully")

    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
    except Exception as e:
        logger.error("Evaluation process failed: %s", e)
        raise


if __name__ == "__main__":
    main()
