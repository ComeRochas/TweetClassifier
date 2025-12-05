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
    
    
features = [
    "extended_tweet",
    "extended_tweet.entities",
    "extended_tweet.entities.urls",
    "extended_tweet.entities.hashtags",
    "extended_tweet.entities.hashtags[].indices",
    "extended_tweet.entities.hashtags[].text",
    "extended_tweet.entities.user_mentions",
    "extended_tweet.entities.symbols",
    "extended_tweet.full_text",
    "extended_tweet.display_text_range",
    "quoted_status",
    "quoted_status.extended_tweet",
    "quoted_status.extended_tweet.extended_entities",
    "quoted_status.extended_tweet.extended_entities.media",
    "quoted_status.extended_tweet.extended_entities.media[].display_url",
    "quoted_status.extended_tweet.extended_entities.media[].indices",
    "quoted_status.extended_tweet.extended_entities.media[].sizes",
    "quoted_status.extended_tweet.extended_entities.media[].sizes.small",
    "quoted_status.extended_tweet.extended_entities.media[].sizes.small.w",
    "quoted_status.extended_tweet.extended_entities.media[].sizes.small.h",
    "quoted_status.extended_tweet.extended_entities.media[].sizes.small.resize",
    "quoted_status.extended_tweet.extended_entities.media[].sizes.large",
    "quoted_status.extended_tweet.extended_entities.media[].sizes.large.w",
    "quoted_status.extended_tweet.extended_entities.media[].sizes.large.h",
    "quoted_status.extended_tweet.extended_entities.media[].sizes.large.resize",
    "quoted_status.extended_tweet.extended_entities.media[].sizes.thumb",
    "quoted_status.extended_tweet.extended_entities.media[].sizes.thumb.w",
    "quoted_status.extended_tweet.extended_entities.media[].sizes.thumb.h",
    "quoted_status.extended_tweet.extended_entities.media[].sizes.thumb.resize",
    "quoted_status.extended_tweet.extended_entities.media[].sizes.medium",
    "quoted_status.extended_tweet.extended_entities.media[].sizes.medium.w",
    "quoted_status.extended_tweet.extended_entities.media[].sizes.medium.h",
    "quoted_status.extended_tweet.extended_entities.media[].sizes.medium.resize",
    "quoted_status.extended_tweet.extended_entities.media[].id_str",
    "quoted_status.extended_tweet.extended_entities.media[].expanded_url",
    "quoted_status.extended_tweet.extended_entities.media[].media_url_https",
    "quoted_status.extended_tweet.extended_entities.media[].id",
    "quoted_status.extended_tweet.extended_entities.media[].type",
    "quoted_status.extended_tweet.extended_entities.media[].media_url",
    "quoted_status.extended_tweet.extended_entities.media[].url",
    "quoted_status.extended_tweet.entities",
    "quoted_status.extended_tweet.entities.urls",
    "quoted_status.extended_tweet.entities.urls[].display_url",
    "quoted_status.extended_tweet.entities.urls[].indices",
    "quoted_status.extended_tweet.entities.urls[].expanded_url",
    "quoted_status.extended_tweet.entities.urls[].url",
    "quoted_status.extended_tweet.entities.hashtags",
    "quoted_status.extended_tweet.entities.hashtags[].indices",
    "quoted_status.extended_tweet.entities.hashtags[].text",
    "quoted_status.extended_tweet.entities.media",
    "quoted_status.extended_tweet.entities.media[].display_url",
    "quoted_status.extended_tweet.entities.media[].indices",
    "quoted_status.extended_tweet.entities.media[].sizes",
    "quoted_status.extended_tweet.entities.media[].sizes.small",
    "quoted_status.extended_tweet.entities.media[].sizes.small.w",
    "quoted_status.extended_tweet.entities.media[].sizes.small.h",
    "quoted_status.extended_tweet.entities.media[].sizes.small.resize",
    "quoted_status.extended_tweet.entities.media[].sizes.large",
    "quoted_status.extended_tweet.entities.media[].sizes.large.w",
    "quoted_status.extended_tweet.entities.media[].sizes.large.h",
    "quoted_status.extended_tweet.entities.media[].sizes.large.resize",
    "quoted_status.extended_tweet.entities.media[].sizes.thumb",
    "quoted_status.extended_tweet.entities.media[].sizes.thumb.w",
    "quoted_status.extended_tweet.entities.media[].sizes.thumb.h",
    "quoted_status.extended_tweet.entities.media[].sizes.thumb.resize",
    "quoted_status.extended_tweet.entities.media[].sizes.medium",
    "quoted_status.extended_tweet.entities.media[].sizes.medium.w",
    "quoted_status.extended_tweet.entities.media[].sizes.medium.h",
    "quoted_status.extended_tweet.entities.media[].sizes.medium.resize",
    "quoted_status.extended_tweet.entities.media[].id_str",
    "quoted_status.extended_tweet.entities.media[].expanded_url",
    "quoted_status.extended_tweet.entities.media[].media_url_https",
    "quoted_status.extended_tweet.entities.media[].id",
    "quoted_status.extended_tweet.entities.media[].type",
    "quoted_status.extended_tweet.entities.media[].media_url",
    "quoted_status.extended_tweet.entities.media[].url",
    "quoted_status.extended_tweet.entities.user_mentions",
    "quoted_status.extended_tweet.entities.user_mentions[].indices",
    "quoted_status.extended_tweet.entities.user_mentions[].screen_name",
    "quoted_status.extended_tweet.entities.user_mentions[].id_str",
    "quoted_status.extended_tweet.entities.user_mentions[].name",
    "quoted_status.extended_tweet.entities.user_mentions[].id",
    "quoted_status.extended_tweet.entities.symbols",
    "quoted_status.extended_tweet.full_text",
    "quoted_status.extended_tweet.display_text_range",
    "quoted_status.in_reply_to_status_id_str",
    "quoted_status.in_reply_to_status_id",
    "quoted_status.created_at",
    "quoted_status.in_reply_to_user_id_str",
    "quoted_status.source",
    "quoted_status.retweet_count",
    "quoted_status.retweeted",
    "quoted_status.geo",
    "quoted_status.filter_level",
    "quoted_status.in_reply_to_screen_name",
    "quoted_status.is_quote_status",
    "quoted_status.id_str",
    "quoted_status.in_reply_to_user_id",
    "quoted_status.favorite_count",
    "quoted_status.id",
    "quoted_status.text",
    "quoted_status.place",
    "quoted_status.lang",
    "quoted_status.quote_count",
    "quoted_status.favorited",
    "quoted_status.possibly_sensitive",
    "quoted_status.coordinates",
    "quoted_status.truncated",
    "quoted_status.reply_count",
    "quoted_status.entities",
    "quoted_status.entities.urls",
    "quoted_status.entities.urls[].display_url",
    "quoted_status.entities.urls[].indices",
    "quoted_status.entities.urls[].expanded_url",
    "quoted_status.entities.urls[].url",
    "quoted_status.entities.hashtags",
    "quoted_status.entities.user_mentions",
    "quoted_status.entities.symbols",
    "quoted_status.display_text_range",
    "quoted_status.contributors",
    "quoted_status.user",
    "quoted_status.user.utc_offset",
    "quoted_status.user.friends_count",
    "quoted_status.user.profile_image_url_https",
    "quoted_status.user.listed_count",
    "quoted_status.user.profile_background_image_url",
    "quoted_status.user.default_profile_image",
    "quoted_status.user.favourites_count",
    "quoted_status.user.description",
    "quoted_status.user.created_at",
    "quoted_status.user.is_translator",
    "quoted_status.user.profile_background_image_url_https",
    "quoted_status.user.protected",
    "quoted_status.user.screen_name",
    "quoted_status.user.id_str",
    "quoted_status.user.profile_link_color",
    "quoted_status.user.translator_type",
    "quoted_status.user.id",
    "quoted_status.user.geo_enabled",
    "quoted_status.user.profile_background_color",
    "quoted_status.user.lang",
    "quoted_status.user.profile_sidebar_border_color",
    "quoted_status.user.profile_text_color",
    "quoted_status.user.verified",
    "quoted_status.user.profile_image_url",
    "quoted_status.user.time_zone",
    "quoted_status.user.url",
    "quoted_status.user.contributors_enabled",
    "quoted_status.user.profile_background_tile",
    "quoted_status.user.profile_banner_url",
    "quoted_status.user.statuses_count",
    "quoted_status.user.follow_request_sent",
    "quoted_status.user.followers_count",
    "quoted_status.user.profile_use_background_image",
    "quoted_status.user.default_profile",
    "quoted_status.user.following",
    "quoted_status.user.name",
    "quoted_status.user.location",
    "quoted_status.user.profile_sidebar_fill_color",
    "quoted_status.user.notifications",
    "in_reply_to_status_id_str",
    "in_reply_to_status_id",
    "created_at",
    "in_reply_to_user_id_str",
    "source",
    "quoted_status_id",
    "retweet_count",
    "retweeted",
    "geo",
    "filter_level",
    "in_reply_to_screen_name",
    "is_quote_status",
    "id_str",
    "in_reply_to_user_id",
    "favorite_count",
    "text",
    "place",
    "quoted_status_permalink",
    "quoted_status_permalink.expanded",
    "quoted_status_permalink.display",
    "quoted_status_permalink.url",
    "lang",
    "quote_count",
    "favorited",
    "coordinates",
    "truncated",
    "timestamp_ms",
    "reply_count",
    "entities",
    "entities.urls",
    "entities.urls[].display_url",
    "entities.urls[].indices",
    "entities.urls[].expanded_url",
    "entities.urls[].url",
    "entities.hashtags",
    "entities.user_mentions",
    "entities.symbols",
    "quoted_status_id_str",
    "contributors",
    "user",
    "user.utc_offset",
    "user.profile_image_url_https",
    "user.listed_count",
    "user.profile_background_image_url",
    "user.default_profile_image",
    "user.favourites_count",
    "user.description",
    "user.created_at",
    "user.is_translator",
    "user.profile_background_image_url_https",
    "user.protected",
    "user.profile_link_color",
    "user.translator_type",
    "user.geo_enabled",
    "user.profile_background_color",
    "user.lang",
    "user.profile_sidebar_border_color",
    "user.profile_text_color",
    "user.profile_image_url",
    "user.time_zone",
    "user.url",
    "user.contributors_enabled",
    "user.profile_background_tile",
    "user.profile_banner_url",
    "user.statuses_count",
    "user.follow_request_sent",
    "user.profile_use_background_image",
    "user.default_profile",
    "user.following",
    "user.location",
    "user.profile_sidebar_fill_color",
    "user.notifications",
    "challenge_id",
    "label",
]

def analyze_user_features():
    print("Loading data...")
    data = []
    with open('train.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # Normalize to flat dataframe
    df = pd.json_normalize(data)
    
    # Filter columns based on the provided features list
    target_cols = [col for col in features if col in df.columns]
    label_col = 'label'
    
    print(f"Found {len(target_cols)} features in dataframe from the provided list.")
    
    # Set plot style
    sns.set_theme(style="whitegrid")

    # First pass: Identify valid features (bool or numeric)
    valid_features = []
    
    for col in target_cols:
        feature_name = col
        
        # Check if feature has only a single value
        try:
            if df[col].nunique() <= 1:
                print(f"Skipping feature {feature_name}: Only one unique value")
                continue
        except TypeError:
            print(f"Skipping feature {feature_name}: Unhashable type (likely list or dict)")
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
    plt.savefig("plots/all_features.png")
    print("\nAll plots saved to plots/all_features.png")
    plt.close()

if __name__ == "__main__":
    analyze_user_features()
