# LLM Evaluation Framework

A comprehensive framework for evaluating Large Language Models (LLMs) using RAGAS metrics with Ollama integration. This framework enables automated assessment of RAG (Retrieval-Augmented Generation) systems by measuring answer quality, context relevance, and faithfulness to source material.

## 🎯 Overview

This framework provides:
- **Multi-Judge Evaluation**: Evaluate LLM responses using multiple judge models for comparative analysis
- **RAGAS Metrics**: Comprehensive evaluation using industry-standard RAGAS metrics
- **Statistical Analysis**: Advanced statistical analysis with visualizations via Jupyter notebooks
- **Ollama Integration**: Local LLM inference and embedding generation
- **Flexible Configuration**: YAML-based configuration with environment variable support

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Evaluation Metrics](#-evaluation-metrics)
- [Statistical Analysis](#-statistical-analysis)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### Core Evaluation
- **Multiple Judge Models**: Compare evaluations across different LLM judges (CodeLlama, Mistral, Qwen2.5)
- **RAGAS Metrics Support**:
  - Answer Relevancy
  - Faithfulness
  - Context Precision
  - Context Recall
- **Batch Processing**: Evaluate multiple models and questions in a single run
- **Robust Error Handling**: Automatic retries and graceful degradation
- **Detailed Logging**: Comprehensive logging for debugging and monitoring

### Statistical Analysis (Jupyter Notebook)
- **Descriptive Statistics**: Mean, standard deviation, min, max, count
- **ANOVA Testing**: Statistical significance testing across model groups
- **Effect Size Analysis**: Cohen's d for practical significance
- **Tukey HSD Tests**: Pairwise model comparisons
- **Comprehensive Visualizations**:
  - Box plots and violin plots
  - Heatmaps (mean scores, standard deviation, correlations)
  - Bar charts with error bars
  - Duration analysis plots
- **Model Ranking**: Automatic ranking based on performance metrics
- **Validation Reports**: Data consistency and reliability checks

## 🚀 Installation

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.ai/) installed and running
- Required Ollama models pulled (see Configuration)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/jv-mt/chat_eval.git
cd chat_eval
```

2. **Create and activate virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Pull required Ollama models**:
```bash
# Judge models
ollama pull codellama:latest
ollama pull mistral:v0.3
ollama pull qwen2.5:latest

# Embedding model
ollama pull nomic-embed-text
```

5. **Configure the framework** (see [Configuration](#-configuration))

## ⚡ Quick Start

### 1. Prepare Your Data

Create two CSV files in the `data/` directory:

**`data/results.csv`** - Model outputs to evaluate:
```csv
query,response,model,documents
"What is Python?","Python is a programming language","gpt-4","['Python is a high-level language', 'Python was created by Guido van Rossum']"
```

**`data/ground_truth.csv`** - Reference answers:
```csv
Question,Ground Truth
"What is Python?","Python is a high-level, interpreted programming language known for its simplicity and readability."
```

### 2. Run Evaluation

```bash
python src/eval.py
```

This will:
- Load your data from CSV files
- Evaluate responses using all configured judge models
- Generate detailed metrics for each model/question combination
- Save results to `outputs/evaluation_by_judge.json`

### 3. Analyze Results

Open and run the Jupyter notebook for statistical analysis:

```bash
jupyter notebook notebooks/judge_stats_modular.ipynb
```

The notebook provides:
- Comprehensive statistical analysis
- Interactive visualizations
- Model rankings
- Performance insights

## ⚙️ Configuration

### Main Configuration (`configs/settings.yml`)

```yaml
# Application Configuration
app:
  name: "LLM Evaluation Framework"
  version: "1.0.0"

# Model Configuration
models:
  judge:
    default: "codellama:latest"
    available:
      - "codellama:latest"
      - "mistral:v0.3"
      - "qwen2.5:latest"

  embedding:
    default: "nomic-embed-text"

# Ollama Configuration
ollama:
  host: "http://localhost:11434"
  temperature: 0.0

# File Paths
paths:
  inputs:
    results: "data/results.csv"
    ground_truth: "data/ground_truth.csv"
  outputs:
    evaluation_output: "outputs/evaluation_by_judge.json"

# Evaluation Configuration
evaluation:
  metrics:
    - "AnswerRelevancy"
    - "Faithfulness"
    - "ContextPrecision"
    - "ContextRecall"

  statistics:
    confidence_level: 0.95
    significance_threshold: 0.05
```

### Environment Variables (Optional)

Create a `.env` file for runtime overrides:

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text

# Judge Models (comma-separated)
JUDGE_MODEL_LIST=codellama:latest,mistral:v0.3,qwen2.5:latest
```

### Logging Configuration (`configs/logging_config.yml`)

Customize logging behavior:
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Output destinations (console, file)
- Format strings

## 📖 Usage

### Command Line Interface

```bash
# Use default configuration
python src/eval.py

# Override file paths
python src/eval.py \
  --results data/my_results.csv \
  --ground-truth data/my_ground_truth.csv \
  --output outputs/my_evaluation.json
```

### Python API

```python
from src.eval import EvaluationConfig, run_evaluation

# Initialize configuration
config = EvaluationConfig()

# Customize if needed
config.judge_models = ["codellama:latest", "mistral:v0.3"]
config.results_csv = "data/results.csv"
config.ground_truth_csv = "data/ground_truth.csv"
config.output_file = "outputs/evaluation.json"

# Run evaluation
run_evaluation(config)
```

### Jupyter Notebook Analysis

The `notebooks/judge_stats_modular.ipynb` notebook provides comprehensive analysis:

1. **Data Loading**: Automatically loads evaluation results
2. **Metric Analysis**: Analyzes each metric separately by chat/judge model
3. **Heatmap Analysis**: Visualizes mean scores and variability
4. **Duration Analysis**: Analyzes response times
5. **Statistical Testing**: ANOVA, Tukey HSD, effect sizes
6. **Model Ranking**: Ranks models by performance

**Key Features**:
- Graceful handling of missing data
- Proper variable initialization
- Best practices for path resolution using `pathlib`
- Comprehensive error messages

## 📊 Evaluation Metrics

### RAGAS Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Answer Relevancy** | Measures how relevant the answer is to the question | 0-1 |
| **Faithfulness** | Measures if the answer is faithful to the context | 0-1 |
| **Context Precision** | Measures precision of retrieved context | 0-1 |
| **Context Recall** | Measures recall of retrieved context | 0-1 |

### Output Format

Results are saved in JSON format:

```json
{
  "judge_model": "codellama:latest",
  "timestamp": "2025-11-03T23:27:29.775",
  "total_samples": 12,
  "results": [
    {
      "model": "gpt-4",
      "question": "What is Python?",
      "metrics": [
        {
          "metric": "AnswerRelevancy",
          "score": 0.95
        },
        {
          "metric": "Faithfulness",
          "score": 0.88
        }
      ],
      "metadata": {
        "duration": 2.5,
        "timestamp": "2025-11-03T23:27:32.123"
      }
    }
  ]
}
```

## 📈 Statistical Analysis

The Jupyter notebook (`notebooks/judge_stats_modular.ipynb`) provides advanced statistical analysis:

### Analysis Components

#### 1. Comprehensive Metric Analysis
- **Box Plots**: Score distributions by chat/judge model
- **Bar Charts**: Mean scores with error bars
- **Violin Plots**: Density distributions
- **Summary Statistics**: Mean, std, min, max, count

#### 2. Heatmap Analysis
- **Mean Score Heatmaps**: By model type (judge/chat)
- **Standard Deviation Heatmaps**: Showing variability
- **Correlation Matrices**: Model consistency analysis
- **Variance Analysis**: Across metrics

#### 3. Duration Analysis
- **Response Time Distributions**: By model
- **Average Duration Comparisons**: With statistical insights
- **Performance Timing Analysis**: Identify bottlenecks

#### 4. Statistical Testing
- **ANOVA Tests**: For group differences
- **Tukey HSD**: Pairwise comparisons
- **Effect Size Analysis**: Cohen's d with interpretation
- **Model Ranking**: Performance-based rankings

#### 5. Advanced Analytics
- **Grand Mean Validation**: Data consistency checks
- **Effect Size Interpretation**: Small/medium/large effects
- **Statistical Significance**: P-value testing
- **Insights & Recommendations**: Actionable findings

### Best Practices Implemented

The notebook follows Python best practices:

✅ **Proper Variable Initialization**: All variables initialized before use
✅ **Graceful Error Handling**: Checks for empty data before analysis
✅ **Path Resolution**: Uses `pathlib.Path` for cross-platform compatibility
✅ **Descriptive Naming**: Global variables use descriptive names (`evaluation_data`, `metadata_list`)
✅ **Conditional Execution**: Analysis only runs when data is available
✅ **Clear Error Messages**: Helpful guidance when data is missing

## 📁 Project Structure

```
chat_eval/
├── configs/
│   ├── settings.yml           # Main configuration file
│   └── logging_config.yml     # Logging configuration
├── data/
│   ├── results.csv            # Model outputs to evaluate
│   └── ground_truth.csv       # Reference answers
├── notebooks/
│   └── judge_stats_modular.ipynb  # Statistical analysis notebook
├── outputs/
│   └── evaluation_by_judge.json   # Evaluation results
├── logs/
│   ├── evaluation.log         # General logs
│   └── evaluation_errors.log  # Error logs
├── src/
│   ├── config.py              # Configuration management
│   └── eval.py                # Main evaluation module
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── LICENSE                    # License information
```

### Key Files

#### `src/eval.py`
Main evaluation module containing:
- `EvaluationConfig`: Configuration class
- `load_and_merge_data()`: Data loading and merging
- `load_and_preprocess_data()`: JSON data preprocessing
- `evaluate_dataset_with_judge()`: Core evaluation logic
- `run_evaluation()`: Main entry point

#### `src/config.py`
Configuration management module:
- `ConfigManager`: Singleton configuration manager
- `get()`: Get configuration values using dot notation
- `setup_logging()`: Initialize logging
- `get_logger()`: Get logger instances

#### `notebooks/judge_stats_modular.ipynb`
Comprehensive statistical analysis notebook with:
- Data loading and preprocessing functions
- Statistical analysis functions (ANOVA, Tukey HSD, Cohen's d)
- Visualization functions (heatmaps, box plots, violin plots)
- Model ranking and comparison functions
- Duration analysis functions

## 🔧 Troubleshooting

### Common Issues

#### 1. Ollama Connection Error
```
Error: Could not connect to Ollama at http://localhost:11434
```
**Solution**: Ensure Ollama is running:
```bash
ollama serve
```

#### 2. Model Not Found
```
Error: Model 'codellama:latest' not found
```
**Solution**: Pull the required model:
```bash
ollama pull codellama:latest
```

#### 3. Empty DataFrame Error in Notebook
```
ValueError: zero-size array to reduction operation
```
**Solution**: Ensure evaluation data exists:
1. Run `python src/eval.py` first to generate evaluation results
2. Check that `outputs/evaluation_by_judge.json` exists
3. Re-run the notebook cells

#### 4. Configuration File Not Found
```
⚠ Configuration file not found: configs/settings.yml
```
**Solution**: Ensure you're running from the project root directory:
```bash
cd /path/to/chat_eval
python src/eval.py
```

#### 5. Import Errors in Notebook
```
ModuleNotFoundError: No module named 'src'
```
**Solution**: The notebook automatically adds the parent directory to the path. Ensure you're running the notebook from the `notebooks/` directory.

## 🎓 Examples

### Example 1: Basic Evaluation

```python
from src.eval import EvaluationConfig, run_evaluation

# Use default configuration
config = EvaluationConfig()
run_evaluation(config)
```

### Example 2: Custom Judge Models

```python
from src.eval import EvaluationConfig, run_evaluation

config = EvaluationConfig()
config.judge_models = ["mistral:v0.3", "qwen2.5:latest"]
run_evaluation(config)
```

### Example 3: Custom Metrics

```python
from src.eval import EvaluationConfig, get_evaluation_metrics
from ragas.metrics import AnswerRelevancy, Faithfulness

# Modify configs/settings.yml:
# evaluation:
#   metrics:
#     - "AnswerRelevancy"
#     - "Faithfulness"

config = EvaluationConfig()
run_evaluation(config)
```

### Example 4: Notebook Analysis with Custom Paths

```python
# In Jupyter notebook
from pathlib import Path
from src.eval import EvaluationConfig

config = EvaluationConfig()

# Override paths for custom data location
notebook_dir = Path.cwd()
config.results_csv = str(notebook_dir.parent / "custom_data" / "results.csv")
config.ground_truth_csv = str(notebook_dir.parent / "custom_data" / "ground_truth.csv")
config.output_file = str(notebook_dir.parent / "custom_outputs" / "evaluation.json")

# Load and analyze data
evaluation_data = load_and_preprocess_data(config.output_file)
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Usage Guide](docs/USAGE.md)**: Detailed usage instructions and examples
- **[API Documentation](docs/API.md)**: Complete API reference for all modules
- **[Contributing Guide](docs/CONTRIBUTING.md)**: Guidelines for contributors

### Quick Links

- [How to prepare data](docs/USAGE.md#data-preparation)
- [Advanced configuration](docs/USAGE.md#advanced-configuration)
- [Custom metrics](docs/USAGE.md#adding-new-metrics)
- [API reference](docs/API.md)
- [Coding standards](docs/CONTRIBUTING.md#coding-standards)

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for detailed guidelines.

**Quick Start for Contributors**:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Follow code style**:
   - Use type hints
   - Add docstrings to all functions
   - Follow PEP 8 style guide
   - Use descriptive variable names
   - Format code with Black
4. **Test your changes**:
   - Run evaluation with test data
   - Verify notebook runs without errors
   - Check for proper error handling
   - Run pytest if tests exist
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for complete guidelines.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Jukka Veijanen**

## 🙏 Acknowledgments

- [RAGAS](https://github.com/explodinggradients/ragas) - RAG Assessment framework
- [Ollama](https://ollama.ai/) - Local LLM inference
- [LangChain](https://www.langchain.com/) - LLM application framework

## 📚 Additional Resources

- [RAGAS Documentation](https://docs.ragas.io/)
- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/README.md)
- [LangChain Documentation](https://python.langchain.com/)

---

**Note**: This framework is designed for research and evaluation purposes. For production use, consider additional error handling, monitoring, and security measures.
