# Fair AI Recruitment System for Software Engineering

A comprehensive solution for addressing bias in AI-driven recruitment systems for software engineering positions.

## Project Overview

This project implements a fair AI recruitment system that identifies and mitigates bias in the candidate selection process. It includes:

1. **Synthetic Dataset Generation**: Creates realistic candidate profiles with protected attributes
2. **Bias Detection**: Implements various fairness metrics to identify bias
3. **Bias Mitigation**: Provides multiple techniques to reduce bias in ML models
4. **Fairness Evaluation**: Analyzes model performance across demographic groups
5. **Visualization**: Creates comprehensive visualizations of fairness metrics
6. **Reporting**: Generates detailed fairness analysis reports

## Components

The system consists of several modular components:

- `dataset_generation.py`: Creates synthetic candidate data with various attributes
- `FairRecruitmentModel.py`: Implements models with bias mitigation techniques
- `FairnessVisualizer.py`: Provides visualization tools for fairness metrics
- `FairnessReportGenerator.py`: Creates comprehensive reports on model fairness
- `main.py`: Orchestrates the entire system

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/fair-ai-recruitment.git
cd fair-ai-recruitment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- imblearn

## Usage

Run the main script to execute the full pipeline:

```bash
python main.py
```

This will:
1. Generate a synthetic dataset (if not already present)
2. Train multiple models with different bias mitigation techniques
3. Evaluate model performance and fairness
4. Generate visualizations
5. Create comprehensive reports

The results will be saved to an `output_[timestamp]` directory, including:
- CSV files with model performance and fairness metrics
- Visualizations of fairness-performance tradeoffs
- An HTML report with detailed analysis
- A Markdown report suitable for sharing

## Understanding the Output

### Fairness Metrics

The system evaluates models using several fairness metrics:

- **Demographic Parity**: Ensures equal selection rates across demographic groups
- **Equal Opportunity**: Ensures equal true positive rates across groups
- **Disparate Impact Ratio**: Measures compliance with the 80% rule
- **Intersectional Fairness**: Analyzes bias at the intersection of multiple attributes

### Bias Mitigation Techniques

Several bias mitigation approaches are implemented:

- **Reweighing**: Adjusts sample weights to balance training data
- **Sampling**: Uses techniques like SMOTE to create balanced datasets
- **Threshold Optimization**: Adjusts decision thresholds for different groups
- **Calibration**: Improves probability estimates for better fairness

### Visualizations

The system generates various visualizations:

- **Performance-Fairness Tradeoff**: Shows how models balance accuracy and fairness
- **Selection Rate Comparison**: Compares selection rates across demographic groups
- **Bias Mitigation Effectiveness**: Visualizes how different techniques reduce bias
- **Feature Importance Analysis**: Shows how different features influence predictions
- **Intersectional Analysis**: Examines bias at the intersection of multiple attributes

## Customization

### Dataset Parameters

You can customize the synthetic dataset by modifying parameters in `dataset_generation.py`:

- Demographic distributions
- Skill correlations
- Bias factors
- Education and experience parameters

### Model Selection

To add new model types, modify the model definitions in `main.py`:

```python
models.append(
    FairRecruitmentModel(
        name="Your New Model",
        model_class=YourModelClass,
        params={'param1': value1},
        bias_mitigation='your_technique'
    )
)
```

### Fairness Constraints

Adjust fairness thresholds in `FairRecruitmentModel.py`:

```python
# Example: Modify threshold for demographic parity
fairness_threshold = 0.05  # Maximum acceptable difference in selection rates
```

## Research Background

This system implements state-of-the-art fairness approaches from the ML research literature:

1. **Pre-processing techniques**: Modify training data to reduce bias
2. **In-processing techniques**: Incorporate fairness constraints in model training
3. **Post-processing techniques**: Adjust model outputs to ensure fairness

The implementation is based on research from papers including:
- "Fairness Through Awareness" (Dwork et al.)
- "Equality of Opportunity in Supervised Learning" (Hardt et al.)
- "Data Preprocessing Techniques for Classification without Discrimination" (Kamiran & Calders)

## Ethical Considerations

When using this system, consider:

1. **Legal compliance**: Ensure your approach complies with relevant employment laws
2. **Stakeholder involvement**: Include diverse perspectives in system design
3. **Transparency**: Be clear about how the system is used in decision-making
4. **Regular auditing**: Continuously monitor for bias and unexpected effects
5. **Human oversight**: Maintain appropriate human review of algorithmic decisions


## Contributors

Eshwar Thota - Initial work

## Acknowledgments

- This project draws on fairness definitions and approaches from the algorithmic fairness research community
- The synthetic data generation approach is inspired by work on realistic dataset generation for fairness research
