# Usage Guide

This guide provides detailed instructions on how to use the LLM Evaluation Framework.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Advanced Configuration](#advanced-configuration)
- [Data Preparation](#data-preparation)
- [Running Evaluations](#running-evaluations)
- [Analyzing Results](#analyzing-results)
- [Customization](#customization)

## Basic Usage

### 1. Prepare Your Data

The framework requires two CSV files:

#### Results CSV (`data/results.csv`)

Contains the model outputs you want to evaluate:

```csv
query,response,model,documents
"What is Python?","Python is a programming language","gpt-4","['Python is a high-level language', 'Python was created by Guido']"
"What is machine learning?","ML is a subset of AI","claude-3","['Machine learning uses algorithms', 'ML enables computers to learn']"
```

**Required Columns**:
- `query`: The input question/prompt
- `response`: The model's generated answer
- `model`: Name of the model that generated the response
- `documents`: Retrieved context documents (as a string representation of a list)

#### Ground Truth CSV (`data/ground_truth.csv`)

Contains reference answers for evaluation:

```csv
Question,Ground Truth
"What is Python?","Python is a high-level, interpreted programming language."
"What is machine learning?","Machine learning is a subset of artificial intelligence."
```

**Required Columns**:
- `Question`: The question (must match `query` in results.csv)
- `Ground Truth`: The reference/correct answer

### 2. Run Evaluation

```bash
# From project root
python src/eval.py
```

This will:
1. Load configuration from `configs/settings.yml`
2. Read data from CSV files
3. Evaluate using all configured judge models
4. Save results to `outputs/evaluation_by_judge.json`

### 3. View Results

Results are saved in JSON format:

```bash
cat outputs/evaluation_by_judge.json | jq
```

### 4. Analyze with Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/judge_stats_modular.ipynb
# Run all cells
```

## Advanced Configuration

### Using Environment Variables

Create a `.env` file in the project root:

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text

# Judge Models (comma-separated)
JUDGE_MODEL_LIST=codellama:latest,mistral:v0.3

# File Paths (optional)
RESULTS_CSV=data/results.csv
GROUND_TRUTH_CSV=data/ground_truth.csv
OUTPUT_FILE=outputs/evaluation.json
```

### Customizing Configuration File

Edit `configs/settings.yml`:

```yaml
# Add or remove judge models
models:
  judge:
    available:
      - "codellama:latest"
      - "mistral:v0.3"
      - "qwen2.5:latest"
      - "llama3:latest"  # Add new model

# Customize metrics
evaluation:
  metrics:
    - "AnswerRelevancy"
    - "Faithfulness"
    - "ContextPrecision"
    - "ContextRecall"
    # Add more metrics as needed

# Adjust statistical thresholds
evaluation:
  statistics:
    confidence_level: 0.95
    significance_threshold: 0.05
```

### Command Line Arguments

Override configuration via command line:

```bash
python src/eval.py \
  --results data/custom_results.csv \
  --ground-truth data/custom_ground_truth.csv \
  --output outputs/custom_evaluation.json
```

## Data Preparation

### Document Format

The `documents` column should contain a string representation of a list:

```python
# Correct format
'["Document 1 text", "Document 2 text"]'

# Also acceptable
"['Document 1 text', 'Document 2 text']"
```

### Handling Special Characters

If your data contains special characters, ensure proper escaping:

```csv
query,response,model,documents
"What is ""Python""?","Python is a language","gpt-4","['Doc 1', 'Doc 2']"
```

### Large Datasets

For large datasets, consider:

1. **Batch Processing**: Split data into smaller chunks
2. **Parallel Evaluation**: Run multiple judge models in parallel
3. **Incremental Results**: Save results after each judge model completes

## Running Evaluations

### Single Judge Model

```python
from src.eval import EvaluationConfig, run_evaluation

config = EvaluationConfig()
config.judge_models = ["codellama:latest"]
run_evaluation(config)
```

### Multiple Judge Models

```python
config = EvaluationConfig()
config.judge_models = [
    "codellama:latest",
    "mistral:v0.3",
    "qwen2.5:latest"
]
run_evaluation(config)
```

### Custom Metrics

Modify `configs/settings.yml` to select specific metrics:

```yaml
evaluation:
  metrics:
    - "AnswerRelevancy"
    - "Faithfulness"
    # Comment out metrics you don't need
    # - "ContextPrecision"
    # - "ContextRecall"
```

### Error Handling

The framework includes automatic retry logic:

```yaml
evaluation:
  timeout: 300        # Timeout per evaluation (seconds)
  max_retries: 3      # Number of retries on failure
```

## Analyzing Results

### Loading Results in Python

```python
import json
import pandas as pd

# Load evaluation results
with open('outputs/evaluation_by_judge.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame
records = []
for judge_result in results:
    judge_model = judge_result['judge_model']
    for result in judge_result['results']:
        for metric in result['metrics']:
            records.append({
                'judge_model': judge_model,
                'chat_model': result['model'],
                'question': result['question'],
                'metric': metric['metric'],
                'score': metric['score']
            })

df = pd.DataFrame(records)
print(df.head())
```

### Jupyter Notebook Analysis

The notebook provides:

1. **Metric Analysis**: Analyze each metric separately
2. **Model Comparison**: Compare performance across models
3. **Statistical Tests**: ANOVA, Tukey HSD, effect sizes
4. **Visualizations**: Heatmaps, box plots, violin plots
5. **Rankings**: Automatic model ranking

### Custom Analysis

Add custom analysis cells to the notebook:

```python
# Example: Filter by specific metric
answer_relevancy = evaluation_data[
    evaluation_data['metric'] == 'AnswerRelevancy'
]

# Calculate statistics
print(answer_relevancy.groupby('chat_model')['score'].describe())

# Create custom visualization
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.boxplot(data=answer_relevancy, x='chat_model', y='score')
plt.title('Answer Relevancy by Chat Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Customization

### Adding New Metrics

1. Import the metric in `src/eval.py`:

```python
from ragas.metrics import YourNewMetric
```

2. Add to metric map in `get_evaluation_metrics()`:

```python
metric_map = {
    "AnswerRelevancy": AnswerRelevancy,
    "YourNewMetric": YourNewMetric,
    # ...
}
```

3. Update `configs/settings.yml`:

```yaml
evaluation:
  metrics:
    - "AnswerRelevancy"
    - "YourNewMetric"
```

### Custom Visualization

Add custom visualization functions to the notebook:

```python
def custom_metric_plot(df, metric_name):
    """Create custom visualization for a specific metric."""
    metric_data = df[df['metric'] == metric_name]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Distribution by judge model
    sns.violinplot(
        data=metric_data,
        x='judge_model',
        y='score',
        ax=axes[0]
    )
    axes[0].set_title(f'{metric_name} - Judge Model Distribution')
    
    # Plot 2: Distribution by chat model
    sns.violinplot(
        data=metric_data,
        x='chat_model',
        y='score',
        ax=axes[1]
    )
    axes[1].set_title(f'{metric_name} - Chat Model Distribution')
    
    plt.tight_layout()
    plt.show()

# Use the function
custom_metric_plot(evaluation_data, 'AnswerRelevancy')
```

### Custom Judge Models

To use a custom Ollama model:

1. Pull the model:
```bash
ollama pull your-custom-model:tag
```

2. Add to configuration:
```yaml
models:
  judge:
    available:
      - "your-custom-model:tag"
```

3. Run evaluation:
```bash
python src/eval.py
```

## Best Practices

1. **Version Control**: Keep track of configuration changes
2. **Data Backup**: Backup evaluation results regularly
3. **Incremental Analysis**: Analyze results after each evaluation run
4. **Documentation**: Document custom metrics and analysis methods
5. **Reproducibility**: Use fixed random seeds and version-controlled configs

## Troubleshooting

See the main [README.md](../README.md#-troubleshooting) for common issues and solutions.

