# Univ.-Dropouts-Statistical-Analysis

# Academic Statistical Analysis

This repository contains scripts for comprehensive statistical analysis of academic data. The main functionality is designed to understand factors associated with student dropout, student profiles, and academic performance.

## Features

- **Descriptive Statistics**: Generate detailed statistical summaries of numerical and categorical variables
- **Statistical Tests**: Perform t-tests, chi-squared tests, and other significance tests
- **Data Visualization**: Create informative visualizations of key academic metrics
- **Predictive Modeling**: Implement logistic regression for dropout prediction
- **Dimensionality Reduction**: Apply PCA to identify key factors
- **Automated Reporting**: Generate comprehensive HTML and CSV reports

## Getting Started

### Installation

You can install this package in two ways:

1. **Using pip directly from GitHub:**
```bash
pip install git+https://github.com/atilamferreira/academic_analysis.git
```

2. **Local installation:**
```bash
git clone https://github.com/atilamferreira/academic_analysis.git
cd academic_analysis
pip install -e .
```

### Prerequisites

The following Python packages are required:
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scipy
- scikit-learn
- plotly

You can install them with:

```bash
pip install -r requirements.txt
```

### Data Format

The script expects CSV files with academic data. Example files are included:
- `dados_exemplo_computacao_200.csv`: Contains anonymized student data for computing courses (200 records)
- `dados_exemplo_energias_200.csv`: Contains anonymized student data for energy-related courses (200 records)

These files contain the following structure:
- Academic information (course, entry method, exit method)
- Performance metrics (grades, completion rates)
- Demographic information

### Running the Analysis

#### Basic Analysis

To run the basic analysis on a single dataset:

```bash
python academic_analysis.py
```

#### Comparative Analysis

To compare two academic datasets (e.g., different courses):

```bash
python comparative_analysis.py
```

This will generate comparisons between computing and energy courses datasets, including:
- Statistical comparisons of demographic profiles
- Academic performance metrics
- Dropout risk factors

#### Specialized Energy and Computing Analysis

For specialized analysis tailored to these specific courses:

```bash
python analise_computacao_energias.py
```

Results for all scripts will be saved in the `result_dados_exemplo` directory, including:
- Visualizations of key metrics
- Detailed HTML reports

## Repository Structure

- `academic_analysis.py`: Main analysis script for single dataset analysis
- `comparative_analysis.py`: Script for comparing multiple academic datasets
- `dados_exemplo_computacao_200.csv`: Example dataset for computing courses (200 records)
- `dados_exemplo_energias_200.csv`: Example dataset for energy courses (200 records)
- `requirements.txt`: Required Python packages

## Citation Requirement

**Important**: Any use, modification, or distribution of this code must include attribution to:

> BEZERRA, Francisco William Coelho. **ANÁLISE DA EVASÃO NOS CURSOS DE ENGENHARIAS DE UMA INSTITUIÇÃO FEDERAL NO INTERIOR DO CEARÁ**. 2025. Dissertação (Mestrado) - Universidade da Integração Internacional da Lusofonia Afro-Brasileira (UNILAB), Ceará, 2025.

This attribution requirement is also specified in the LICENSE file.

## License

This project is licensed under the Apache License 2.0 with an attribution requirement - see the LICENSE file for details.

## Acknowledgments

This analysis framework was developed to help educational institutions better understand and address student dropout, with the goal of improving retention rates and academic outcomes. 
