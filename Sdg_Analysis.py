#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

# --- Configuration --- #
DATA_DIR = 'data'
FILE_2023 = os.path.join(DATA_DIR, 'sustainable_development_report_2023_original.csv')
FILE_2000_2022 = os.path.join(DATA_DIR, 'sdg_index_2000-2022_original.csv')

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Created data directory: {DATA_DIR}")

# --- Data Loading and Preprocessing Functions --- #

def load_and_preprocess_data(file_2023: str, file_2000_2022: str) -> pd.DataFrame:
    """
    Loads and preprocesses the SDG index data from two separate CSV files.

    Args:
        file_2023 (str): Path to the 2023 SDG report data.
        file_2000_2022 (str): Path to the 2000-2022 SDG index data.

    Returns:
        pd.DataFrame: A merged and cleaned DataFrame containing SDG data.
    """
    print(f"Loading 2023 data from {file_2023}")
    df_2023 = pd.read_csv(file_2023)
    print(f"Loading 2000-2022 data from {file_2000_2022}")
    df_2000_2022 = pd.read_csv(file_2000_2022)

    # Preprocess 2023 data
    if 'region' in df_2023.columns:
        df_2023 = df_2023.drop('region', axis=1)
    df_2023.insert(2, 'year', 2023) # Insert year as integer
    df_2023.rename(columns={'overall_score': 'sdg_index_score'}, inplace=True)

    # Merge dataframes
    merged_df = pd.concat([df_2000_2022, df_2023], ignore_index=True)

    # Sort and clean country names
    merged_df = merged_df.sort_values(by=["country_code", "year"]).reset_index(drop=True)
    merged_df["country"] = merged_df["country"].str.title()

    # Define region mapping (original dictionary from the notebook)
    region_mapping = {
        "Northern Africa": [
            "Algeria", "Egypt", "Libya", "Morocco", "Sudan", "Tunisia", "Western Sahara"
        ],
        "Eastern Africa": [
            "British Indian Ocean Territory", "Burundi", "Comoros", "Djibouti", "Eritrea",
            "Ethiopia", "French Southern Territories", "Kenya", "Madagascar", "Malawi",
            "Mauritius", "Mayotte", "Mozambique", "Réunion", "Rwanda", "Seychelles",
            "Somalia", "South Sudan", "Uganda", "United Republic of Tanzania", "Zambia", "Zimbabwe"
        ],
        "Middle Africa": [
            "Angola", "Cameroon", "Central African Republic", "Chad", "Congo", "Democratic Republic of the Congo",
            "Equatorial Guinea", "Gabon", "Sao Tome and Principe"
        ],
        "Southern Africa": [
            "Botswana", "Eswatini", "Lesotho", "Namibia", "South Africa"
        ],
        "Western Africa": [
            "Benin", "Burkina Faso", "Cabo Verde", "Côte d’Ivoire", "Gambia", "Ghana", "Guinea", "Guinea-Bissau",
            "Liberia", "Mali", "Mauritania", "Niger", "Nigeria", "Saint Helena", "Senegal", "Sierra Leone", "Togo"
        ],
        "Caribbean": [
            "Anguilla", "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados", "Bonaire", "Sint Eustatius and Saba",
            "British Virgin Islands", "Cayman Islands", "Cuba", "Curaçao", "Dominica", "Dominican Republic", "Grenada",
            "Guadeloupe", "Haiti", "Jamaica", "Martinique", "Montserrat", "Puerto Rico", "Saint Barthélemy",
            "Saint Kitts and Nevis", "Saint Lucia", "Saint Martin (French Part)", "Saint Vincent and the Grenadines",
            "Sint Maarten (Dutch part)", "Trinidad and Tobago", "Turks and Caicos Islands", "United States Virgin Islands"
        ],
        "Central America": [
            "Belize", "Costa Rica", "El Salvador", "Guatemala", "Honduras", "Mexico", "Nicaragua", "Panama"
        ],
        "South America": [
            "Argentina", "Bolivia", "Bouvet Island", "Brazil", "Chile", "Colombia", "Ecuador", "Falkland Islands (Malvinas)",
            "French Guiana", "Guyana", "Paraguay", "Peru", "South Georgia and the South Sandwich Islands", "Suriname",
            "Uruguay", "Venezuela"
        ],
        "Northern America": [
            "Bermuda", "Canada", "Greenland", "Saint Pierre and Miquelon", "United States of America"
        ],
        "Central Asia": [
            "Kazakhstan", "Kyrgyzstan", "Tajikistan", "Turkmenistan", "Uzbekistan"
        ],
        "Eastern Asia": [
            "China", "China - Hong Kong Special Administrative Region", "China - Macao Special Administrative Region",
            "Democratic People\'s Republic of Korea", "Japan", "Mongolia", "Republic of Korea"
        ],
        "Southeastern Asia": [
            "Brunei Darussalam", "Cambodia", "Indonesia", "Lao People\'s Democratic Republic", "Malaysia", "Myanmar",
            "Philippines", "Singapore", "Thailand", "Timor-Leste", "Vietnam"
        ],
        "Southern Asia": [
            "Afghanistan", "Bangladesh", "Bhutan", "India", "Iran", "(Islamic Republic of)", "Maldives", "Nepal",
            "Pakistan", "Sri Lanka"
        ],
        "Western Asia": [
            "Armenia", "Azerbaijan", "Bahrain", "Cyprus", "Georgia", "Iraq", "Israel", "Jordan", "Kuwait", "Lebanon",
            "Oman", "Qatar", "Saudi Arabia", "State of Palestine", "Syrian Arab Republic", "Turkey", "United Arab Emirates",
            "Yemen"
        ],
        "Eastern Europe": [
            "Belarus", "Bulgaria", "Czech Republic", "Hungary", "Poland", "Republic of Moldova", "Romania", "Russian Federation",
            "Slovakia", "Ukraine"
        ],
        "Northern Europe": [
            "Åland Islands", "Channel Islands (Guernsey, Jersey, Sark)", "Denmark", "Estonia", "Faroe Islands", "Finland",
            "Iceland", "Ireland", "Isle of Man", "Latvia", "Lithuania", "Norway", "Svalbard and Jan Mayen Islands",
            "Sweden", "United Kingdom of Great Britain and Northern Ireland"
        ],
        "Southern Europe": [
            "Albania", "Andorra", "Bosnia and Herzegovina", "Croatia", "Gibraltar", "Greece", "Holy See", "Italy", "Malta",
            "Montenegro", "Portugal", "San Marino", "Serbia", "Slovenia", "Spain", "The former Yugoslav Republic of Macedonia"
        ],
        "Western Europe": [
            "Austria", "Belgium", "France", "Germany", "Liechtenstein", "Luxembourg", "Monaco", "Netherlands", "Switzerland"
        ],
        "Australia and New Zealand": [
            "Australia", "Christmas Island", "Cocos (Keeling) Islands", "Heard Island and McDonald Islands", "New Zealand", "Norfolk Island"
        ],
        "Melanesia": [
            "Fiji", "New Caledonia", "Papua New Guinea", "Solomon Islands", "Vanuatu"
        ],
        "Micronesia": [
            "Guam", "Kiribati", "Marshall Islands", "Micronesia (Federated States of)", "Nauru", "Northern Mariana Islands",
            "Palau", "United States Minor Outlying Islands"
        ]
    }

    country_to_region = {
        country: region
        for region, countries in region_mapping.items()
        for country in countries
    }

    merged_df["region"] = merged_df["country"].map(lambda x: country_to_region.get(x, "Unknown"))

    # Standardize country names for better mapping
    country_name_replacements = {
        'Bahamas, The': 'Bahamas',
        'Bosnia And Herzegovina': 'Bosnia and Herzegovina',
        "Cote D\'Ivoire": 'Côte d’Ivoire',
        'Congo, Dem. Rep.': 'Democratic Republic of the Congo',
        'Congo, Rep.': 'Congo',
        'Czechia': 'Czech Republic',
        'Egypt, Arab Rep.': 'Egypt',
        'United Kingdom': 'United Kingdom of Great Britain and Northern Ireland',
        'Gambia, The': 'Gambia',
        'Iran, Islamic Rep.': 'Iran',
        'Kyrgyz Republic': 'Kyrgyzstan',
        'Korea, Rep.': 'Republic of Korea',
        'Lao Pdr': "Lao People\'s Democratic Republic",
        'Moldova': 'Republic of Moldova',
        'North Macedonia': 'The former Yugoslav Republic of Macedonia',
        'Sao Tome And Principe': 'Sao Tome and Principe',
        'Slovak Republic': 'Slovakia',
        'Trinidad And Tobago': 'Trinidad and Tobago',
        'Türkiye': 'Turkey',
        'Tanzania': 'United Republic of Tanzania',
        'United States': 'United States of America',
        'Venezuela, Rb': 'Venezuela',
        'Yemen, Rep.': 'Yemen'
    }
    merged_df['country'] = merged_df['country'].replace(country_name_replacements)

    # Update regions for newly standardized country names
    merged_df.loc[merged_df["region"] == "Unknown", "region"] = merged_df["country"].map(country_to_region)

    # Remove aggregate rows (original notebook had these)
    aggregate_countries = [
        "Sub-Saharan Africa", "Eastern Europe And Central Asia", "East And South Asia",
        "High-Income Countries", "Latin America And The Caribbean", "Low-Income Countries",
        "Lower & Lower-Middle Income", "Lower-Middle-Income Countries", "Middle East And North Africa",
        "Oecd Members", "Oceania", "Small Island Developing States", "Upper-Middle-Income Countries", "World"
    ]
    merged_df_clean = merged_df[~merged_df['country'].isin(aggregate_countries)].copy()

    # Reorder columns to place 'region' earlier
    cols = merged_df_clean.columns.tolist()
    if 'region' in cols:
        cols.insert(2, cols.pop(cols.index('region')))
    merged_df_clean = merged_df_clean[cols]

    print("Data loading and preprocessing complete.")
    return merged_df_clean


def plot_sdg_score_by_region(df: pd.DataFrame, output_path: str = 'sdg_score_by_region.png'):
    """
    Generates and saves a line plot of SDG index score by region over time.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        output_path (str): Path to save the plot image.
    """
    data_grouped = df.groupby(['region', 'year'])['sdg_index_score'].mean().reset_index()

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=data_grouped, x='year', y='sdg_index_score', hue='region', linewidth=1.5, marker='o', markersize=4)
    plt.title('Average SDG Index Score by Region (2000-2023)', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average SDG Index Score', fontsize=12)
    plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_correlation_heatmap(df: pd.DataFrame, output_path: str = 'sdg_correlation_heatmap.png'):
    """
    Generates and saves a correlation heatmap of SDG scores.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        output_path (str): Path to save the plot image.
    """
    # Assuming SDG scores are from column index 5 to 21 (goal_1_score to goal_17_score)
    sdg_scores_cols = [col for col in df.columns if 'goal_' in col and '_score' in col]
    sdg_scores = df[sdg_scores_cols]

    correlation_matrix = sdg_scores.corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=.5)
    plt.title('Correlation Heatmap of Individual SDG Scores', fontsize=16)
    plt.xlabel('SDG Indicators', fontsize=12)
    plt.ylabel('SDG Indicators', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_sdg_regression(df: pd.DataFrame, sdg_x: str, sdg_y: str, output_path: str = 'sdg_regression_plot.png'):
    """
    Generates and saves a regression plot between two specified SDG scores.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        sdg_x (str): Column name for the x-axis SDG score (e.g., 'goal_3_score').
        sdg_y (str): Column name for the y-axis SDG score (e.g., 'goal_16_score').
        output_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    sns.regplot(data=df, x=sdg_x, y=sdg_y, scatter_kws={'alpha':0.6, 'color': 'skyblue'}, line_kws={'color': 'red', 'linewidth': 2})

    plt.xlabel(f'{sdg_x.replace("_", " ").title()} Score', fontsize=12)
    plt.ylabel(f'{sdg_y.replace("_", " ").title()} Score', fontsize=12)
    plt.title(f'Regression Analysis: {sdg_x.replace("_", " ").title()} vs. {sdg_y.replace("_", " ").title()}', fontsize=16)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_top_bottom_countries(df: pd.DataFrame, sdg_column: str = 'sdg_index_score', num_countries: int = 5, output_path: str = 'top_bottom_sdg_countries.png'):
    """
    Generates and saves a bar plot showing top and bottom performing countries based on an SDG score.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        sdg_column (str): The SDG score column to analyze.
        num_countries (int): Number of top and bottom countries to display.
        output_path (str): Path to save the plot image.
    """
    country_avg_sdg = df.groupby('country')[sdg_column].mean().reset_index()

    top_countries = country_avg_sdg.nlargest(num_countries, sdg_column)
    bottom_countries = country_avg_sdg.nsmallest(num_countries, sdg_column)

    top_countries['Performance Group'] = 'Top Performers'
    bottom_countries['Performance Group'] = 'Bottom Performers'

    top_bottom_countries = pd.concat([top_countries, bottom_countries])

    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=top_bottom_countries,
        x=sdg_column,
        y='country',
        hue='Performance Group',
        palette={'Top Performers': 'forestgreen', 'Bottom Performers': 'firebrick'},
        dodge=False
    )
    plt.title(f'Top {num_countries} and Bottom {num_countries} Countries by Average {sdg_column.replace("_", " ").title()}', fontsize=16)
    plt.xlabel(f'Average {sdg_column.replace("_", " ").title()}', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_sdg_correlation_with_index(df: pd.DataFrame, output_path: str = 'sdg_index_correlation.png'):
    """
    Calculates and visualizes the correlation between individual SDG scores and the overall SDG Index Score.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        output_path (str): Path to save the plot image.
    """
    sdg_columns = [col for col in df.columns if 'goal_' in col and '_score' in col]
    
    if 'sdg_index_score' not in df.columns:
        print("Error: 'sdg_index_score' column not found in DataFrame.")
        return

    # Calculate the correlation between each SDG score and the SDG index score
    # Ensure all columns exist before attempting correlation
    relevant_cols = [col for col in sdg_columns + ['sdg_index_score'] if col in df.columns]
    if len(relevant_cols) < 2:
        print("Not enough relevant columns to calculate correlation.")
        return

    correlations = df[relevant_cols].corr()

    # Extract the correlation between SDG index and each SDG score
    correlation_with_index = correlations['sdg_index_score'].drop('sdg_index_score', errors='ignore')

    # Sort the correlations in descending order
    sorted_correlations = correlation_with_index.sort_values(ascending=False)

    plt.figure(figsize=(12, 7))
    sns.barplot(x=sorted_correlations.index, y=sorted_correlations.values, palette='viridis')
    plt.title('Correlation Between Individual SDG Goals and Overall SDG Index Score', fontsize=16)
    plt.xlabel('Individual SDG Goal', fontsize=12)
    plt.ylabel('Correlation Coefficient with SDG Index', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")




def plot_interactive_sdg_progress(df: pd.DataFrame, output_html_path: str = 'plots/interactive_sdg_progress.html'):
    """
    Generates and saves an interactive line plot of SDG index score by region over time using Plotly.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        output_html_path (str): Path to save the interactive HTML plot.
    """
    region_data = df.groupby(['region', 'year'])['sdg_index_score'].mean().reset_index()

    fig = px.line(region_data,
                  x='year',
                  y='sdg_index_score',
                  color='region',
                  title='Interactive SDG Progress by Region',
                  labels={'sdg_index_score': 'Average SDG Index Score', 'year': 'Year'},
                  hover_name='region')
    fig.update_layout(hovermode="x unified")
    fig.write_html(output_html_path)
    print(f"Interactive plot saved to {output_html_path}")


# Update main function to include the new plot
def main():
    """
    Main function to execute the SDG data analysis workflow.
    """
    # Load and preprocess data
    merged_clean_df = load_and_preprocess_data(FILE_2023, FILE_2000_2022)

    # Generate plots
    plot_sdg_score_by_region(merged_clean_df, output_path=\'plots/sdg_score_by_region.png\')
    plot_correlation_heatmap(merged_clean_df, output_path=\'plots/sdg_correlation_heatmap.png\')
    plot_sdg_regression(merged_clean_df, \'goal_3_score\', \'goal_16_score\', output_path=\'plots/sdg_3_vs_sdg_16_regression.png\')
    plot_top_bottom_countries(merged_clean_df, output_path=\'plots/top_bottom_sdg_countries.png\')
    plot_sdg_correlation_with_index(merged_clean_df, output_path=\'plots/sdg_index_correlation.png\')
    plot_interactive_sdg_progress(merged_clean_df, output_html_path=\'plots/interactive_sdg_progress.html\')

    print("All analysis and plotting complete. Check the \'plots\' directory for output.")

if __name__ == "__main__":
    # Create plots directory if it doesn\'t exist
    if not os.path.exists(\'plots\'):
        os.makedirs(\'plots\')
    main()

