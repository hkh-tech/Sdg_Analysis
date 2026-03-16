# Sustainable Development Goals (SDG) Index Analysis

This project was developed as part of the Data Science course at Lusófona University. It was a collaborative effort between myself and my colleague, Nina Petrushkova.

## Project Overview

Our project analyzes Sustainable Development Goals (SDG) Index scores from 2000 to 2023. We focused on identifying trends, correlations between different SDG goals, and highlighting country performance using data from the Sustainable Development Report. The analysis is implemented in Python, utilizing libraries such as Pandas, Matplotlib, Seaborn, and Plotly for data processing and visualization.

## Key Features

-   **Data Integration & Preprocessing**: Combines historical and recent SDG data, standardizing country and regional information.
-   **Trend Analysis**: Visualizes regional SDG Index score progression over time.
-   **Correlation Insights**: Explores relationships between various SDG goals through correlation heatmaps and regression plots.
-   **Performance Benchmarking**: Identifies top and bottom performing countries based on average SDG Index scores.
-   **Interactive Visualizations**: Provides interactive plots for dynamic exploration of regional SDG progress.

## Setup and Usage

1.  **Dependencies**: Install required Python libraries:
    ```bash
    pip install pandas numpy seaborn matplotlib plotly
    ```
2.  **Data Placement**: Place `sustainable_development_report_2023_original.csv` and `sdg_index_2000-2022_original.csv` into a `data/` directory in the project root.
3.  **Execution**: Run the main script:
    ```bash
    python enhanced_sdg_analysis.py
    ```

## Output

Upon execution, the script generates various static image plots and an interactive HTML plot in a `plots/` directory, summarizing the SDG analysis.

## Contributing

This project was a collaborative effort. We welcome further contributions and suggestions.
