# Jupyter Notebook Guide

This guide provides detailed information about the statistical analysis notebook (`notebooks/judge_stats_modular.ipynb`).

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Notebook Structure](#notebook-structure)
- [Analysis Components](#analysis-components)
- [Customization](#customization)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The `judge_stats_modular.ipynb` notebook provides comprehensive statistical analysis and visualization of RAGAS evaluation results. It processes evaluation data from multiple judge models and chat models, computing descriptive statistics, performing ANOVA tests, and generating various visualizations.

### Key Features

- **Automated Data Loading**: Loads evaluation results from JSON files
- **Comprehensive Statistics**: Mean, std, min, max, count for all metrics
- **Statistical Testing**: ANOVA, Tukey HSD, effect size analysis
- **Rich Visualizations**: Heatmaps, box plots, violin plots, bar charts
- **Model Ranking**: Automatic ranking based on performance
- **Duration Analysis**: Response time analysis and insights
- **Error Handling**: Graceful handling of missing data

## Getting Started

### Prerequisites

1. **Run Evaluation First**: Generate evaluation results
   ```bash
   python src/eval.py
   ```

2. **Verify Data Exists**: Check that output file exists
   ```bash
   ls -lh outputs/evaluation_by_judge.json
   ```

3. **Start Jupyter**: Launch Jupyter notebook
   ```bash
   jupyter notebook
   ```

4. **Open Notebook**: Navigate to `notebooks/judge_stats_modular.ipynb`

### Running the Notebook

1. **Run All Cells**: Click "Cell" → "Run All"
2. **Or Run Sequentially**: Execute cells one by one with Shift+Enter
3. **Review Outputs**: Examine visualizations and statistics

## Notebook Structure

### Cell 1: Imports and Configuration

```python
# Imports all required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
# ... and more

# Loads configuration
from src.eval import EvaluationConfig
config = EvaluationConfig()

# Configures metrics to analyze
metrics_to_evaluate = [
    "AnswerRelevancy",
    "ContextPrecision",
    "ContextRecall",
    "Faithfulness",
]
```

**What it does**:
- Imports all necessary libraries
- Sets up configuration with proper path resolution
- Defines metrics to analyze
- Initializes logging

### Cell 2-4: Utility Functions

Helper functions for:
- Setting centered x-axis ticks
- Adding value labels to bars
- Parsing metadata from JSON strings

### Cell 5-8: Data Processing Functions

Functions for:
- Loading and preprocessing evaluation data
- Merging results with ground truth
- Parsing document contexts
- Creating RAGAS datasets

### Cell 9: Data Loading

```python
# Initialize data structures
evaluation_data = pd.DataFrame(columns=[...])
metadata_list = []

# Load evaluation data
try:
    evaluation_data = load_and_preprocess_data(config.output_file)
    # ... filter and process
except FileNotFoundError:
    print("Error: Evaluation output file not found")
```

**What it does**:
- Initializes variables with proper empty values
- Loads evaluation results from JSON
- Loads and parses metadata for duration analysis
- Handles missing files gracefully

### Cell 10: Metric Analysis

```python
if not evaluation_data.empty:
    # Analyze each metric separately
    for metric in unique_metrics:
        visualize_metric_analysis(evaluation_data, metric, group_col="chat_model")
else:
    print("⚠️  Skipping metric analysis - evaluation data not loaded")
```

**What it does**:
- Analyzes each metric by chat model
- Analyzes each metric by judge model
- Creates comprehensive visualizations
- Shows statistical summaries

**Outputs**:
- Box plots showing score distributions
- Bar charts with error bars
- Violin plots for density visualization
- Summary statistics tables

### Cell 11: Heatmap Analysis

```python
if not evaluation_data.empty:
    visualize_heatmap_analysis(evaluation_data, "judge")
    visualize_heatmap_analysis(evaluation_data, "chat")
else:
    print("⚠️  Skipping heatmap analysis - evaluation data not loaded")
```

**What it does**:
- Creates mean score heatmaps
- Creates standard deviation heatmaps
- Generates correlation matrices
- Performs variance analysis

**Outputs**:
- 2x2 grid of heatmaps for each model type
- Statistical insights and interpretations

### Cell 12: Tukey HSD Tests

```python
if not evaluation_data.empty:
    tukey = pairwise_tukeyhsd(evaluation_data["score"], evaluation_data["chat_model"])
    print(tukey)
else:
    print("⚠️  Skipping Tukey HSD test - evaluation data not loaded")
```

**What it does**:
- Performs pairwise comparisons between models
- Tests for statistical significance
- Provides confidence intervals

**Outputs**:
- Tukey HSD test results table
- Significance indicators

### Cell 13: Effect Size Analysis

```python
if not evaluation_data.empty:
    effect_sizes = calculate_pairwise_effect_sizes(evaluation_data, unique_models)
    print_effect_sizes(effect_sizes)
else:
    print("⚠️  Skipping effect size analysis - evaluation data not loaded")
```

**What it does**:
- Calculates Cohen's d for all model pairs
- Interprets effect sizes (small/medium/large)
- Identifies practically significant differences

**Outputs**:
- Effect size matrix
- Interpretation guide

## Analysis Components

### 1. Descriptive Statistics

**Function**: `compute_descriptive_stats()`

**Computes**:
- Mean scores
- Standard deviation
- Minimum and maximum values
- Sample counts

**Usage**:
```python
stats = compute_descriptive_stats(evaluation_data, "chat_model", metrics_to_evaluate)
print(stats)
```

### 2. ANOVA Testing

**Function**: `perform_anova()`

**Tests**:
- Null hypothesis: All groups have equal means
- Alternative: At least one group differs

**Interpretation**:
- p < 0.05: Significant difference exists
- p ≥ 0.05: No significant difference

**Usage**:
```python
anova_results = perform_anova(evaluation_data, "chat_model", metrics_to_evaluate)
for metric, (f_stat, p_value) in anova_results.items():
    print(f"{metric}: F={f_stat:.3f}, p={p_value:.4f}")
```

### 3. Model Ranking

**Function**: `rank_models()`

**Ranks models by**:
- Overall average performance
- Per-metric performance

**Usage**:
```python
overall_ranks, per_metric_ranks = rank_models(evaluation_data, metrics_to_evaluate)
print(overall_ranks)
```

### 4. Visualizations

#### Box Plots
- Show distribution of scores
- Identify outliers
- Compare medians and quartiles

#### Violin Plots
- Show probability density
- Reveal distribution shape
- Compare across groups

#### Heatmaps
- Visualize mean scores
- Show variability (std)
- Identify patterns

#### Bar Charts
- Compare mean scores
- Show error bars (std)
- Easy comparison across groups

## Customization

### Adding Custom Metrics

1. **Add to metrics list**:
```python
metrics_to_evaluate = [
    "AnswerRelevancy",
    "Faithfulness",
    "YourCustomMetric",  # Add here
]
```

2. **Ensure data contains the metric**:
```python
print(evaluation_data['metric'].unique())
```

### Custom Visualizations

Add a new cell with custom plotting code:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Filter data for specific metric
metric_data = evaluation_data[evaluation_data['metric'] == 'AnswerRelevancy']

# Create custom plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.swarmplot(data=metric_data, x='chat_model', y='score', ax=ax)
ax.set_title('Answer Relevancy Score Distribution')
ax.set_xlabel('Chat Model')
ax.set_ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Custom Statistical Tests

Add custom analysis:

```python
from scipy.stats import mannwhitneyu

# Compare two specific models
model1_scores = evaluation_data[
    evaluation_data['chat_model'] == 'gpt-4'
]['score']
model2_scores = evaluation_data[
    evaluation_data['chat_model'] == 'claude-3'
]['score']

# Perform Mann-Whitney U test
statistic, p_value = mannwhitneyu(model1_scores, model2_scores)
print(f"Mann-Whitney U: statistic={statistic}, p={p_value}")
```

### Filtering Data

Analyze specific subsets:

```python
# Filter by judge model
codellama_data = evaluation_data[
    evaluation_data['judge_model'] == 'codellama:latest'
]

# Filter by metric
faithfulness_data = evaluation_data[
    evaluation_data['metric'] == 'Faithfulness'
]

# Filter by score threshold
high_scores = evaluation_data[evaluation_data['score'] >= 0.8]

# Analyze filtered data
visualize_metric_analysis(high_scores, 'AnswerRelevancy', 'chat_model')
```

## Best Practices

### 1. Run Cells Sequentially

Always run cells in order from top to bottom to ensure:
- Variables are properly initialized
- Dependencies are loaded
- Data is available for analysis

### 2. Clear Outputs Before Committing

```bash
# In Jupyter: Cell → All Output → Clear
# Or use nbconvert
jupyter nbconvert --clear-output --inplace notebooks/judge_stats_modular.ipynb
```

### 3. Save Visualizations

```python
# Save figure to file
fig.savefig('outputs/figures/metric_analysis.png', dpi=300, bbox_inches='tight')
```

### 4. Document Custom Analysis

Add markdown cells to explain custom analysis:

```markdown
## Custom Analysis: Model Comparison

This section compares GPT-4 and Claude-3 on the AnswerRelevancy metric
using a Mann-Whitney U test to determine if there's a statistically
significant difference in their performance.
```

### 5. Handle Missing Data

Always check for missing data:

```python
# Check for missing values
print(evaluation_data.isnull().sum())

# Handle missing scores
evaluation_data = evaluation_data.dropna(subset=['score'])
```

## Troubleshooting

### Issue: "Evaluation data not loaded"

**Cause**: Evaluation results file doesn't exist

**Solution**:
```bash
# Run evaluation first
python src/eval.py

# Verify file exists
ls -lh outputs/evaluation_by_judge.json
```

### Issue: "ValueError: zero-size array"

**Cause**: Trying to analyze empty DataFrame

**Solution**: The notebook now includes proper checks. If you see this error:
1. Ensure evaluation data was loaded successfully
2. Check that the data contains the metrics you're analyzing
3. Verify the data isn't filtered to empty

### Issue: "ModuleNotFoundError: No module named 'src'"

**Cause**: Python path not set correctly

**Solution**: The notebook automatically adds parent directory to path. If issue persists:
```python
import sys
import os
sys.path.append(os.path.abspath(".."))
```

### Issue: Plots not displaying

**Cause**: Matplotlib backend issue

**Solution**:
```python
%matplotlib inline
import matplotlib.pyplot as plt
```

### Issue: Out of memory

**Cause**: Large dataset or too many visualizations

**Solution**:
```python
# Clear figures after displaying
plt.close('all')

# Reduce figure size
default_figsize = (10, 4)  # Instead of (15, 6)

# Process data in chunks
for metric in metrics_to_evaluate:
    # Analyze one metric at a time
    # Clear memory after each
```

## Additional Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [SciPy Stats Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)

---

For more information, see the [main documentation](../README.md).

