import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import math

# Create directory for plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

def analyze_user_features():
    print("Loading data...")
    data = []
    with open('train.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # Normalize to flat dataframe
    df = pd.json_normalize(data)
    
    # Filter columns that start with 'user.'
    user_cols = [col for col in df.columns if col.startswith('user.')]
    label_col = 'label'
    
    print(f"Found {len(user_cols)} user features.")
    
    # Set plot style
    sns.set_theme(style="whitegrid")

    # First pass: Identify valid features (bool or numeric)
    valid_features = []
    
    for col in user_cols:
        feature_name = col.replace('user.', '')
        
        # Check if feature has only a single value
        if df[col].nunique() <= 1:
            print(f"Skipping feature {feature_name}: Only one unique value")
            continue

        # Determine type
        is_bool = False
        is_numeric = False
        
        # Check type
        if pd.api.types.is_bool_dtype(df[col]):
            is_bool = True
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Check if it looks like a boolean (only 0s and 1s)
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                pass # Treat as numeric unless explicitly bool type, logic kept from original
            is_numeric = True
        elif df[col].dtype == 'object':
            # Check if it contains boolean values
            unique_vals = set(df[col].dropna().unique())
            if unique_vals.issubset({True, False}):
                is_bool = True
        
        if is_bool:
            valid_features.append({'col': col, 'name': feature_name, 'type': 'bool'})
        elif is_numeric:
            valid_features.append({'col': col, 'name': feature_name, 'type': 'numeric'})
        else:
            print(f"Skipping feature {feature_name}: Type Other ({df[col].dtype})")

    # Calculate grid size based on valid features only
    n_features = len(valid_features)
    if n_features == 0:
        print("No valid features to plot.")
        return

    n_cols = 3  # Number of columns in the grid
    n_rows = math.ceil(n_features / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten() # Flatten to 1D array for easy iteration

    # Second pass: Generate plots
    for i, feature_info in enumerate(valid_features):
        ax = axes[i]
        col = feature_info['col']
        feature_name = feature_info['name']
        feat_type = feature_info['type']

        print(f"\n{'='*60}")
        print(f"Analyzing feature: {feature_name}")
        print(f"{'='*60}")
        
        # Count nulls
        num_nulls = df[col].isnull().sum()
        print(f"Number of null values: {num_nulls}")

        if feat_type == 'bool':
            print(f"Type: Boolean")
            # Crosstab for counts
            crosstab = pd.crosstab(df[col].fillna('NaN'), df[label_col])
            print("\nDistribution (Count per Label):")
            print(crosstab)
            
            # Visualization
            sns.countplot(x=col, hue=label_col, data=df, ax=ax)
            ax.set_title(f"Distribution of {feature_name}")
            ax.set_xlabel(feature_name)
            ax.set_ylabel("Count")
            
            # Add numerical values over the bars
            for container in ax.containers:
                ax.bar_label(container)

        elif feat_type == 'numeric':
            print(f"Type: Numeric")
            
            # Median and Quartiles per label
            stats = df.groupby(label_col)[col].describe(percentiles=[.25, .5, .75])
            print("\nStatistics per label (Median and Quartiles):")
            print(stats[['25%', '50%', '75%']])
            
            # Visualization
            # Check range of data for log scale
            data_range = df[col].max() - df[col].min()
            use_log = False
            if data_range > 1000 and df[col].min() >= 0:
                use_log = True
            
            try:
                sns.boxplot(x=label_col, y=col, data=df, ax=ax)
                ax.set_title(f"Boxplot of {feature_name}")
                if use_log:
                    ax.set_yscale('symlog')
                    ax.set_ylabel(f"{feature_name} (log scale)")
            except Exception as e:
                print(f"Could not plot {feature_name}: {e}")
                ax.text(0.5, 0.5, f"Error plotting {feature_name}", ha='center', va='center')

    # Hide empty subplots if any (remaining slots in the grid)
    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("plots/all_user_features.png")
    print("\nAll plots saved to plots/all_user_features.png")
    plt.close()

if __name__ == "__main__":
    analyze_user_features()
