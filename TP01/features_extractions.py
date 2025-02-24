import pandas as pd
import numpy as np
import re


def load_data(file_paths):
    """
    Load datasets from given file paths.

    Parameters:
    file_paths (dict): Dictionary containing file names as keys and file paths as values.

    Returns:
    dict: Dictionary containing loaded datasets as pandas DataFrames.
    """
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


def features_extraction(df):
    """
    Create new features.
    1. FollowingFollowersRatio: NumberOfFollowings/NumberOfFollowers
    2. AccountLifetime: AccountLifetime difference between CollectedAt and CreatedAt
    3. AverageTweetsPerDay: Average Tweets Per Day

    Parameters:
    df (pd.DataFrame): DataFrame containing profile data.

    Returns:
    pd.DataFrame: DataFrame with new features added.
    """
    df["FollowingFollowersRatio"] = df["NumberOfFollowings"] / df["NumberOfFollowers"]
    df["CreatedAt"] = pd.to_datetime(df["CreatedAt"])
    df["CollectedAt"] = pd.to_datetime(df["CollectedAt"])
    df["AccountLifetime"] = (df["CollectedAt"] - df["CreatedAt"]).dt.days
    df["AverageTweetsPerDay"] = df["NumberOfTweets"] / df["AccountLifetime"]
    return df


def count_urls(text):
    """
    Count URLs in a given text.

    Parameters:
    text (str): Text in which URLs are to be counted.

    Returns:
    int: Number of URLs found in the text.
    """
    urls = re.findall(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        str(text),
    )
    return len(urls)


def calculate_proportion_url(df):
    """
    Process tweet data to create new features.

    Parameters:
    df (pd.DataFrame): DataFrame containing tweet data.

    Returns:
    pd.DataFrame: DataFrame with new features added.
    """
    df["URL_Count"] = df["Tweet"].apply(count_urls)
    df["Proportion_URL"] = df["URL_Count"] / df["Tweet"].notna().astype(int)
    df = calculate_at_proportion(df)
    return df


def calculate_at_proportion(df):
    """
    Calculate the proportion of '@' mentions in tweets.

    Parameters:
    df (pd.DataFrame): DataFrame containing tweet data.

    Returns:
    pd.DataFrame: DataFrame with '@' mention proportion feature added.
    """
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
    """
    Calculate time difference between consecutive tweets.

    Parameters:
    df (pd.DataFrame): DataFrame containing tweet,Creation date and time  data.

    Returns:
    pd.DataFrame: DataFrame with mean and max time difference between tweets for each user in second.
    """
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
    """
    Merge profile and tweet data.

    Parameters:
    profile_df (pd.DataFrame): DataFrame containing profile data.
    tweet_df (pd.DataFrame): DataFrame containing tweet data.
    Here drop all duplicates base on UserID and we keep the first occurence where we have the same index  multiple times

    Returns:
    pd.DataFrame: Merged DataFrame containing profile and tweet data.
    """
    tweet_df_no_duplicates = tweet_df.drop_duplicates(subset="UserID", keep="first")
    profile_df = profile_df.reset_index(drop=True)
    tweet_df_no_duplicates = tweet_df_no_duplicates.reset_index(drop=True)
    merged_df = pd.concat([profile_df, tweet_df_no_duplicates], axis=1)
    return merged_df


def main():
    """
    Main function to load, process, and merge datasets, then save the final dataset as CSV.

    Note: The path of the files must be specified as "Datasets/content_polluters.text" for the code to give an output
    """
    # Define file paths for datasets
    file_paths = {
        "polluters_profile": "Datasets/content_polluters.txt",
        "polluters_followings": "Datasets/content_polluters_followings.txt",
        "polluters_tweets": "Datasets/content_polluters_tweets.txt",
        "legitimate_profile": "Datasets/legitimate_users.txt",
        "legitimate_followings": "Datasets/legitimate_users_followings.txt",
        "legitimate_tweets": "Datasets/legitimate_users_tweets.txt",
    }

    # Load data from file paths
    data = load_data(file_paths)

    # Process profile data
    data["polluters_profile"] = features_extraction(data["polluters_profile"])
    data["legitimate_profile"] = features_extraction(data["legitimate_profile"])

    # Process tweet data
    data["polluters_tweets"] = calculate_proportion_url(data["polluters_tweets"])
    data["legitimate_tweets"] = calculate_proportion_url(data["legitimate_tweets"])

    # Calculate time differences between consecutive tweets
    polluters_time_diff = calculate_time_diff(data["polluters_tweets"])
    legitimate_time_diff = calculate_time_diff(data["legitimate_tweets"])

    # Merge profile data with time differences
    data["polluters_profile"] = data["polluters_profile"].merge(
        polluters_time_diff, on="UserID", how="left"
    )
    data["legitimate_profile"] = data["legitimate_profile"].merge(
        legitimate_time_diff, on="UserID", how="left"
    )

    # Final merging of profile and tweet data
    polluters_final = merge_data(data["polluters_profile"], data["polluters_tweets"])
    legitimate_final = merge_data(data["legitimate_profile"], data["legitimate_tweets"])

    # Add class labels to the datasets
    polluters_final["Class"] = 1
    legitimate_final["Class"] = 0

    # Drop unnecessary columns before concatenation
    columns_to_drop = ["UserID", "CreatedAt", "CollectedAt", "TweetID", "Tweet"]
    polluters_final = polluters_final.drop(columns=columns_to_drop)
    legitimate_final = legitimate_final.drop(columns=columns_to_drop)

    # Combine both datasets and save to CSV
    final_df = pd.concat([polluters_final, legitimate_final], axis=0, ignore_index=True)
    final_df.to_csv("combine_datasets.csv", index=False)
    print("The datasets are ready and the name of the file is: combine_datasets.csv")
    print(final_df.head())


if __name__ == "__main__":
    main()
