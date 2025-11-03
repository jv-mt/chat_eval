# API Documentation

This document provides detailed API documentation for the LLM Evaluation Framework.

## Table of Contents

- [Core Modules](#core-modules)
- [Configuration Module](#configuration-module)
- [Evaluation Module](#evaluation-module)
- [Notebook Functions](#notebook-functions)

## Core Modules

### `src.config`

Configuration management module for loading settings and managing logging.

#### Classes

##### `ConfigManager`

Singleton configuration manager that loads settings from YAML files.

**Attributes**:
- `project_root` (Path): Project root directory
- `config_file` (Path): Path to settings.yml
- `logging_config_file` (Path): Path to logging_config.yml
- `config` (Dict[str, Any]): Configuration dictionary

**Methods**:

```python
def get(key: str, default: Any = None) -> Any:
    """
    Get configuration value using dot notation.
    
    Args:
        key: Configuration key (e.g., 'database.host')
        default: Default value if key not found
    
    Returns:
        Configuration value or default
    
    Example:
        >>> config = ConfigManager()
        >>> host = config.get('ollama.host', 'http://localhost:11434')
    """
```

```python
def setup_logging() -> logging.Logger:
    """
    Setup logging from configs/logging_config.yml.
    
    Returns:
        Logger instance
    
    Example:
        >>> config = ConfigManager()
        >>> logger = config.setup_logging()
        >>> logger.info("Logging configured")
    """
```

```python
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (defaults to module name)
    
    Returns:
        Logger instance
    
    Example:
        >>> logger = config.get_logger(__name__)
    """
```

#### Module-Level Functions

```python
def get(key: str, default: Any = None) -> Any:
    """
    Get configuration value using dot notation.
    
    Args:
        key: Configuration key (e.g., 'app.name')
        default: Default value if key not found
    
    Returns:
        Configuration value or default
    
    Example:
        >>> from src.config import get
        >>> app_name = get('app.name', 'Unknown')
    """
```

```python
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    
    Example:
        >>> from src.config import get_logger
        >>> logger = get_logger(__name__)
    """
```

```python
def setup_logging() -> logging.Logger:
    """
    Setup logging configuration.
    
    Returns:
        Logger instance
    
    Example:
        >>> from src.config import setup_logging
        >>> logger = setup_logging()
    """
```

### `src.eval`

Main evaluation module for running RAGAS evaluations with Ollama.

#### Classes

##### `EvaluationConfig`

Configuration class for evaluation settings.

**Attributes**:
- `ollama_host` (str): Ollama server URL
- `embedding_model` (str): Embedding model name
- `judge_models` (List[str]): List of judge model names
- `results_csv` (str): Path to results CSV file
- `ground_truth_csv` (str): Path to ground truth CSV file
- `output_file` (str): Path to output JSON file
- `temperature` (float): LLM temperature setting
- `timeout` (int): Evaluation timeout in seconds
- `max_retries` (int): Maximum retry attempts

**Example**:
```python
from src.eval import EvaluationConfig

config = EvaluationConfig()
print(f"Judge models: {config.judge_models}")
print(f"Output file: {config.output_file}")
```

#### Functions

##### `get_evaluation_metrics()`

```python
def get_evaluation_metrics() -> List[Any]:
    """
    Get the list of RAGAS metrics to use for evaluation.
    
    Returns:
        List of RAGAS metric classes from configuration or defaults.
    
    Example:
        >>> metrics = get_evaluation_metrics()
        >>> print([m.__name__ for m in metrics])
        ['AnswerRelevancy', 'Faithfulness', 'ContextPrecision', 'ContextRecall']
    """
```

##### `parse_document_contexts()`

```python
def parse_document_contexts(documents: pd.Series) -> List[List[str]]:
    """
    Parse document strings into context lists for RAGAS evaluation.
    
    Args:
        documents: Pandas Series containing document strings
    
    Returns:
        List of context lists, where each inner list contains
        up to 2 document strings.
    
    Example:
        >>> docs = pd.Series(['["doc1", "doc2"]', '["doc3", "doc4"]'])
        >>> contexts = parse_document_contexts(docs)
        >>> print(contexts)
        [['doc1', 'doc2'], ['doc3', 'doc4']]
    """
```

##### `load_and_merge_data()`

```python
def load_and_merge_data(config: EvaluationConfig) -> pd.DataFrame:
    """
    Load and merge evaluation data from CSV files.
    
    Args:
        config: Configuration instance containing file paths
    
    Returns:
        Merged DataFrame containing columns from both files
    
    Raises:
        FileNotFoundError: If CSV files don't exist
    
    Example:
        >>> config = EvaluationConfig()
        >>> df = load_and_merge_data(config)
        >>> print(df.columns)
        Index(['query', 'response', 'model', 'documents', 'Ground Truth'])
    """
```

##### `load_and_preprocess_data()`

```python
def load_and_preprocess_data(json_file: str) -> pd.DataFrame:
    """
    Load and preprocess evaluation data from JSON file.
    
    Args:
        json_file: Path to JSON file containing evaluation results
    
    Returns:
        DataFrame with columns: judge_model, timestamp, chat_model,
        question, metric, score
    
    Raises:
        FileNotFoundError: If JSON file doesn't exist
    
    Example:
        >>> df = load_and_preprocess_data('outputs/evaluation_by_judge.json')
        >>> print(df.head())
    """
```

##### `create_ragas_dataset()`

```python
def create_ragas_dataset(df: pd.DataFrame) -> Dataset:
    """
    Create RAGAS dataset from DataFrame.
    
    Args:
        df: DataFrame with columns: query, response, documents, Ground Truth
    
    Returns:
        RAGAS Dataset object ready for evaluation
    
    Example:
        >>> df = load_and_merge_data(config)
        >>> dataset = create_ragas_dataset(df)
        >>> print(len(dataset))
        12
    """
```

##### `evaluate_dataset_with_judge()`

```python
def evaluate_dataset_with_judge(
    dataset: Dataset,
    config: EvaluationConfig,
    judge_model: str
) -> List[Dict[str, Any]]:
    """
    Evaluate dataset using a specific judge model.
    
    Args:
        dataset: RAGAS dataset to evaluate
        config: Configuration instance
        judge_model: Name of the judge model to use
    
    Returns:
        List of evaluation results with metrics and scores
    
    Example:
        >>> results = evaluate_dataset_with_judge(
        ...     dataset, config, "codellama:latest"
        ... )
        >>> print(len(results))
        12
    """
```

##### `save_evaluation_results()`

```python
def save_evaluation_results(
    results: List[Dict[str, Any]],
    config: EvaluationConfig
) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: List of evaluation results
        config: Configuration instance with output file path
    
    Example:
        >>> save_evaluation_results(results, config)
        >>> # Results saved to outputs/evaluation_by_judge.json
    """
```

##### `run_evaluation()`

```python
def run_evaluation(config: EvaluationConfig) -> None:
    """
    Main entry point for running evaluation.
    
    Args:
        config: Configuration instance
    
    Example:
        >>> config = EvaluationConfig()
        >>> run_evaluation(config)
        # Runs evaluation with all configured judge models
    """
```

## Notebook Functions

The Jupyter notebook (`notebooks/judge_stats_modular.ipynb`) contains numerous analysis functions.

### Data Processing Functions

##### `parse_metadata()`

```python
def parse_metadata(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Parse JSON metadata strings from DataFrame.
    
    Args:
        df: DataFrame with metadata column
    
    Returns:
        List of parsed metadata dictionaries
    """
```

### Statistical Analysis Functions

##### `compute_descriptive_stats()`

```python
def compute_descriptive_stats(
    df: pd.DataFrame,
    group_col: str,
    metrics: List[str]
) -> pd.DataFrame:
    """
    Compute descriptive statistics for each metric and group.
    
    Args:
        df: DataFrame with evaluation data
        group_col: Column to group by ('chat_model' or 'judge_model')
        metrics: List of metric names to analyze
    
    Returns:
        DataFrame with mean, std, min, max, count for each group
    """
```

##### `perform_anova()`

```python
def perform_anova(
    df: pd.DataFrame,
    group_col: str,
    metrics: List[str]
) -> Dict[str, Tuple[float, float]]:
    """
    Perform ANOVA test for each metric.
    
    Args:
        df: DataFrame with evaluation data
        group_col: Column to group by
        metrics: List of metric names
    
    Returns:
        Dictionary mapping metric names to (F-statistic, p-value) tuples
    """
```

##### `calculate_pairwise_effect_sizes()`

```python
def calculate_pairwise_effect_sizes(
    df: pd.DataFrame,
    models: List[str],
    score_col: str = "score",
    model_col: str = "chat_model"
) -> Dict[Tuple[str, str], float]:
    """
    Calculate Cohen's d effect sizes for all model pairs.
    
    Args:
        df: DataFrame with evaluation data
        models: List of model names
        score_col: Column containing scores
        model_col: Column containing model names
    
    Returns:
        Dictionary mapping (model1, model2) tuples to Cohen's d values
    """
```

### Visualization Functions

##### `visualize_metric_analysis()`

```python
def visualize_metric_analysis(
    df: pd.DataFrame,
    metric_name: str,
    group_col: str = "chat_model",
    figsize: Tuple[int, int] = (15, 6)
) -> None:
    """
    Create comprehensive visualization for a specific metric.
    
    Args:
        df: DataFrame with evaluation data
        metric_name: Name of the metric to visualize
        group_col: Column to group by
        figsize: Figure size tuple
    """
```

##### `visualize_heatmap_analysis()`

```python
def visualize_heatmap_analysis(
    df: pd.DataFrame,
    model_type: str = "judge",
    figsize: Tuple[int, int] = (15, 6)
) -> None:
    """
    Create comprehensive heatmap analysis for a model type.
    
    Args:
        df: DataFrame with evaluation data
        model_type: 'judge' or 'chat'
        figsize: Figure size tuple
    """
```

### Ranking Functions

##### `rank_models()`

```python
def rank_models(
    df: pd.DataFrame,
    metrics: List[str]
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Rank models based on performance metrics.
    
    Args:
        df: DataFrame with evaluation data
        metrics: List of metric names
    
    Returns:
        Tuple of (overall_rankings, per_metric_rankings)
    """
```

## Usage Examples

### Basic Evaluation

```python
from src.eval import EvaluationConfig, run_evaluation

# Initialize and run
config = EvaluationConfig()
run_evaluation(config)
```

### Custom Configuration

```python
from src.eval import EvaluationConfig, run_evaluation
from pathlib import Path

config = EvaluationConfig()
config.judge_models = ["codellama:latest"]
config.results_csv = "data/custom_results.csv"
config.output_file = "outputs/custom_output.json"

run_evaluation(config)
```

### Loading and Analyzing Results

```python
from src.eval import load_and_preprocess_data
import pandas as pd

# Load results
df = load_and_preprocess_data('outputs/evaluation_by_judge.json')

# Analyze
print(df.groupby(['metric', 'chat_model'])['score'].mean())
```

## Error Handling

All functions include proper error handling and logging:

```python
from src.config import get_logger

logger = get_logger(__name__)

try:
    run_evaluation(config)
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
except Exception as e:
    logger.error(f"Evaluation failed: {e}")
```

