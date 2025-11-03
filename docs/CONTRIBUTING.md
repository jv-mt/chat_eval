# Contributing to LLM Evaluation Framework

Thank you for your interest in contributing to the LLM Evaluation Framework! This document provides guidelines and best practices for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Documentation](#documentation)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior

- Be respectful and considerate
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Acknowledge different viewpoints and experiences

### Unacceptable Behavior

- Harassment or discriminatory language
- Personal attacks or trolling
- Publishing others' private information
- Other conduct that could be considered inappropriate

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Git
- Ollama installed and running
- Basic understanding of:
  - Python programming
  - RAGAS metrics
  - Statistical analysis
  - Jupyter notebooks

### Finding Issues to Work On

1. Check the [Issues](../../issues) page
2. Look for issues labeled `good first issue` or `help wanted`
3. Comment on the issue to express interest
4. Wait for maintainer approval before starting work

### Proposing New Features

1. Open an issue describing the feature
2. Explain the use case and benefits
3. Discuss implementation approach
4. Wait for feedback before implementing

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/jv-mt/chat_eval.git
cd chat_eval
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Development Dependencies

```bash
pip install -r requirements-dev.txt  # If available
# Or install manually:
pip install pytest black flake8 mypy
```

### 5. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

## Coding Standards

### Python Style Guide

Follow [PEP 8](https://pep8.org/) style guide:

- Use 4 spaces for indentation (no tabs)
- Maximum line length: 88 characters (Black default)
- Use descriptive variable names
- Add docstrings to all functions and classes

### Code Formatting

Use [Black](https://black.readthedocs.io/) for code formatting:

```bash
# Format all Python files
black src/ notebooks/

# Check formatting without making changes
black --check src/
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import List, Dict, Optional, Tuple, Any

def process_data(
    data: pd.DataFrame,
    config: EvaluationConfig,
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Process evaluation data.
    
    Args:
        data: Input DataFrame
        config: Configuration instance
        metrics: Optional list of metrics to process
    
    Returns:
        Dictionary containing processed results
    """
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_statistics(
    df: pd.DataFrame,
    group_col: str,
    metrics: List[str]
) -> pd.DataFrame:
    """
    Calculate descriptive statistics for metrics.
    
    This function computes mean, standard deviation, min, max, and count
    for each metric grouped by the specified column.
    
    Args:
        df: DataFrame containing evaluation data with columns:
            - score: Numerical scores
            - metric: Metric names
            - {group_col}: Grouping column
        group_col: Column name to group by (e.g., 'chat_model')
        metrics: List of metric names to analyze
    
    Returns:
        DataFrame with statistics for each group and metric.
        Columns: {group_col}, metric, mean, std, min, max, count
    
    Raises:
        ValueError: If required columns are missing from DataFrame
        KeyError: If specified metrics don't exist in data
    
    Example:
        >>> df = pd.DataFrame({
        ...     'chat_model': ['gpt-4', 'gpt-4', 'claude'],
        ...     'metric': ['AnswerRelevancy', 'Faithfulness', 'AnswerRelevancy'],
        ...     'score': [0.95, 0.88, 0.92]
        ... })
        >>> stats = calculate_statistics(df, 'chat_model', ['AnswerRelevancy'])
        >>> print(stats)
    
    Note:
        Missing values are automatically excluded from calculations.
    """
    pass
```

### Variable Naming Conventions

#### Global Variables

Use descriptive names for global variables:

```python
# Good
evaluation_data = pd.DataFrame()
metadata_list = []
configuration_settings = {}

# Bad
df = pd.DataFrame()
data = []
config = {}
```

#### Function Parameters

Use conventional short names for function parameters:

```python
# Good - short names in local scope
def analyze_data(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pass

# Also good - descriptive names when needed
def merge_datasets(
    results_dataframe: pd.DataFrame,
    ground_truth_dataframe: pd.DataFrame
) -> pd.DataFrame:
    pass
```

#### Constants

Use UPPER_CASE for constants:

```python
DEFAULT_FIGSIZE = (15, 6)
MAX_RETRIES = 3
TIMEOUT_SECONDS = 300
```

### Error Handling

Always include proper error handling:

```python
from src.config import get_logger

logger = get_logger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Empty file: {file_path}")
        raise ValueError(f"File is empty: {file_path}")
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise
```

### Logging

Use the logging module instead of print statements:

```python
from src.config import get_logger

logger = get_logger(__name__)

# Good
logger.info("Starting evaluation")
logger.debug(f"Processing {len(data)} records")
logger.warning("Missing optional configuration")
logger.error(f"Failed to load file: {e}")

# Bad
print("Starting evaluation")
print(f"Processing {len(data)} records")
```

## Testing Guidelines

### Writing Tests

Create tests for all new functionality:

```python
import pytest
import pandas as pd
from src.eval import load_and_merge_data, EvaluationConfig

def test_load_and_merge_data():
    """Test data loading and merging."""
    config = EvaluationConfig()
    config.results_csv = "tests/data/test_results.csv"
    config.ground_truth_csv = "tests/data/test_ground_truth.csv"
    
    df = load_and_merge_data(config)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'query' in df.columns
    assert 'Ground Truth' in df.columns

def test_load_missing_file():
    """Test error handling for missing files."""
    config = EvaluationConfig()
    config.results_csv = "nonexistent.csv"
    
    with pytest.raises(FileNotFoundError):
        load_and_merge_data(config)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_eval.py

# Run specific test
pytest tests/test_eval.py::test_load_and_merge_data
```

### Test Coverage

Aim for at least 80% code coverage for new code.

## Pull Request Process

### 1. Prepare Your Changes

```bash
# Ensure code is formatted
black src/ notebooks/

# Run linter
flake8 src/

# Run type checker
mypy src/

# Run tests
pytest
```

### 2. Commit Your Changes

Use clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Add support for custom RAGAS metrics"
git commit -m "Fix: Handle empty DataFrame in heatmap analysis"
git commit -m "Docs: Update API documentation for new functions"

# Bad commit messages
git commit -m "Update code"
git commit -m "Fix bug"
git commit -m "Changes"
```

### 3. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 4. Create Pull Request

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill in the PR template:
   - Description of changes
   - Related issues
   - Testing performed
   - Screenshots (if applicable)

### 5. Code Review

- Address reviewer feedback promptly
- Make requested changes in new commits
- Don't force-push after review has started
- Be open to suggestions and constructive criticism

### 6. Merge

Once approved, a maintainer will merge your PR.

## Documentation

### Code Documentation

- Add docstrings to all functions and classes
- Include type hints
- Provide usage examples in docstrings
- Document complex algorithms

### README Updates

Update README.md if you:
- Add new features
- Change configuration options
- Modify installation process
- Add new dependencies

### API Documentation

Update `docs/API.md` for:
- New functions or classes
- Changed function signatures
- New modules

### Usage Guide

Update `docs/USAGE.md` for:
- New usage patterns
- Configuration changes
- New examples

### Jupyter Notebook

For notebook changes:
- Clear all outputs before committing
- Add markdown cells explaining analysis
- Include example outputs in documentation
- Test notebook from clean state

## Best Practices

### Path Handling

Use `pathlib.Path` for cross-platform compatibility:

```python
from pathlib import Path

# Good
data_dir = Path("data")
results_file = data_dir / "results.csv"

# Bad
results_file = "data/results.csv"  # Won't work on Windows
```

### Configuration

Use the configuration system:

```python
from src.config import get

# Good
timeout = get("evaluation.timeout", 300)

# Bad
timeout = 300  # Hardcoded value
```

### Data Validation

Validate inputs:

```python
def process_scores(scores: List[float]) -> float:
    """Calculate mean score."""
    if not scores:
        raise ValueError("Scores list cannot be empty")
    
    if not all(0 <= s <= 1 for s in scores):
        raise ValueError("All scores must be between 0 and 1")
    
    return sum(scores) / len(scores)
```

## Questions?

If you have questions:
1. Check existing documentation
2. Search closed issues
3. Open a new issue with the `question` label
4. Join our community discussions

Thank you for contributing! 🎉

