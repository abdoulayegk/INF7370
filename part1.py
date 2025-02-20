import pandas as pd
import numpy as np
import re
import polars as pl


def load_data(file_paths):
    """Load datasets from given file paths."""
    datasets = {}
    for name, path in file_paths.items():
        if "profile" in name:
            datasets[name] = pd.read_table(
                path,
                header=None,
                names=[
                    "UserID",
                    "CreatedAt",
                    "CollectedAt",
                    "NumberOfFollowings",
                    "NumberOfFollowers",
                    "NumberOfTweets",
                    "LengthOfScreenName",
                    "LengthOfDescriptionInUserProfile",
                ],
            )
        elif "followings" in name:
            datasets[name] = pd.read_table(
                path,
                header=None,
                names=["UserID", "SeriesOfNumberOfFollowings"],
            )
        elif "tweets" in name:
            datasets[name] = pd.read_table(
                path,
                header=None,
                names=["UserID", "TweetID", "Tweet", "CreatedAt"],
            )
    return datasets


def process_profile_data(df):
    """Process profile data to create new features."""
    df["FollowingFollowersRatio"] = df["NumberOfFollowings"] / df["NumberOfFollowers"]
    df["CreatedAt"] = pd.to_datetime(df["CreatedAt"])
    df["CollectedAt"] = pd.to_datetime(df["CollectedAt"])
    df["AccountLifetime"] = (df["CollectedAt"] - df["CreatedAt"]).dt.days
    df["AverageTweetsPerDay"] = df["NumberOfTweets"] / df["AccountLifetime"]
    return df


def count_urls(text):
    """Count URLs in a given text."""
    urls = re.findall(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        str(text),
    )
    return len(urls)


def process_tweet_data(df):
    """Process tweet data to create new features."""
    df["URL_Count"] = df["Tweet"].apply(count_urls)
    df["Proportion_URL"] = df["URL_Count"] / df["Tweet"].notna().astype(int)
    df = calculate_at_proportion(df)
    return df


def calculate_at_proportion(df):
    """Calculate the proportion of '@' mentions in tweets."""
    df["Tweet"] = df["Tweet"].astype(str)
    df["count_AT"] = df["Tweet"].apply(lambda x: x.count("@"))
    df["Tweet_Length"] = df["Tweet"].apply(len)
    df["Proportion_AT"] = df.apply(
        lambda row: (
            row["count_AT"] / row["Tweet_Length"] if row["Tweet_Length"] != 0 else 0
        ),
        axis=1,
    )
    return df


def calculate_time_diff(df):
    """Calculate time difference between consecutive tweets."""
    df["CreatedAt"] = pd.to_datetime(df["CreatedAt"])
    df = df.sort_values(by=["UserID", "CreatedAt"])
    df["Time_Diff"] = df.groupby("UserID")["CreatedAt"].diff()
    df["Time_Diff_Seconds"] = df["Time_Diff"].dt.total_seconds()
    time_stats = (
        df.groupby("UserID")["Time_Diff_Seconds"].agg(["mean", "max"]).reset_index()
    )
    time_stats.columns = [
        "UserID",
        "Mean_Time_Between_Tweets",
        "Max_Time_Between_Tweets",
    ]
    return time_stats


def merge_data(profile_df, tweet_df):
    """Merge profile and tweet data."""
    tweet_df_no_duplicates = tweet_df.drop_duplicates(subset="UserID", keep="first")
    profile_df = profile_df.reset_index(drop=True)
    tweet_df_no_duplicates = tweet_df_no_duplicates.reset_index(drop=True)
    merged_df = pd.concat([profile_df, tweet_df_no_duplicates], axis=1)
    return merged_df


def drop_unecessary_features(df):
    df = df.drop()
    return df


def main():
    file_paths = {
        "polluters_profile": "Datasets/content_polluters.txt",
        "polluters_followings": "Datasets/content_polluters_followings.txt",
        "polluters_tweets": "Datasets/content_polluters_tweets.txt",
        "legitimate_profile": "Datasets/legitimate_users.txt",
        "legitimate_followings": "Datasets/legitimate_users_followings.txt",
        "legitimate_tweets": "Datasets/legitimate_users_tweets.txt",
    }

    data = load_data(file_paths)

    # Process profile data
    data["polluters_profile"] = process_profile_data(data["polluters_profile"])
    data["legitimate_profile"] = process_profile_data(data["legitimate_profile"])

    # Process tweet data
    data["polluters_tweets"] = process_tweet_data(data["polluters_tweets"])
    data["legitimate_tweets"] = process_tweet_data(data["legitimate_tweets"])

    # Calculate time differences
    polluters_time_diff = calculate_time_diff(data["polluters_tweets"])
    legitimate_time_diff = calculate_time_diff(data["legitimate_tweets"])

    # Merge data
    data["polluters_profile"] = data["polluters_profile"].merge(
        polluters_time_diff, on="UserID", how="left"
    )
    data["legitimate_profile"] = data["legitimate_profile"].merge(
        legitimate_time_diff, on="UserID", how="left"
    )

    # Final merging and processing
    polluters_final = merge_data(data["polluters_profile"], data["polluters_tweets"])
    legitimate_final = merge_data(data["legitimate_profile"], data["legitimate_tweets"])

    # Add class labels
    polluters_final["Class"] = 1
    legitimate_final["Class"] = 0

    # Combine datasets and save to CSV
    final_df = pd.concat([polluters_final, legitimate_final], axis=0, ignore_index=True)
    final_df.to_csv("Combined_datasets_backup.csv", index=False)


if __name__ == "__main__":
    main()
