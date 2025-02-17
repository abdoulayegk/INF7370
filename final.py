import pandas as pd
import numpy as np
import re
import polars as pl


def read_and_process_files():
    # Read content polluters profile data
    polluters_profile = pd.read_table(
        "Datasets/content_polluters.txt",
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
    """Here we are going to create all the new features that are related to the dataset polluters and we will do the same later for the other one too."""
    polluters_profile["FollowingFollowersRatio"] = (
        polluters_profile["NumberOfFollowings"] / polluters_profile["NumberOfFollowers"]
    )
    # Convert date columns to datetime
    polluters_profile["CreatedAt"] = pd.to_datetime(polluters_profile["CreatedAt"])
    polluters_profile["CollectedAt"] = pd.to_datetime(polluters_profile["CollectedAt"])

    # Calculate account lifetime in days or Durée de vie du compte
    polluters_profile["AccountLifetime"] = (
        polluters_profile["CollectedAt"] - polluters_profile["CreatedAt"]
    ).dt.days

    # Calculate average tweets per day
    polluters_profile["AverageTweetsPerDay"] = (
        polluters_profile["NumberOfTweets"] / polluters_profile["AccountLifetime"]
    )

    # ================================= End of content_polluters==========================================

    # Read content polluters followings data
    polluters_followings = pd.read_table(
        "Datasets/content_polluters_followings.txt",
        header=None,
        names=["UserID", "SeriesOfNumberOfFollowings"],
    )

    # ================================= End of content_polluters_followings==========================================

    # Read content polluters tweets data
    polluters_tweets = pd.read_table(
        "Datasets/content_polluters_tweets.txt",
        header=None,
        names=["UserID", "TweetID", "Tweet", "CreatedAt"],
    )

    # ===================================reading the legitimate datasets=========================================
    # Read legitimate users profile data
    legitimate_profile = pd.read_table(
        "Datasets/legitimate_users.txt",
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
    legitimate_profile["FollowingFollowersRatio"] = (
        legitimate_profile["NumberOfFollowings"]
        / legitimate_profile["NumberOfFollowers"]
    )

    # Convert date columns to datetime
    legitimate_profile["CreatedAt"] = pd.to_datetime(legitimate_profile["CreatedAt"])
    legitimate_profile["CollectedAt"] = pd.to_datetime(
        legitimate_profile["CollectedAt"]
    )

    # Calculate account lifetime in days or Durée de vie du compte
    legitimate_profile["AccountLifetime"] = (
        legitimate_profile["CollectedAt"] - legitimate_profile["CreatedAt"]
    ).dt.days

    # Calculate average tweets per day
    legitimate_profile["AverageTweetsPerDay"] = (
        legitimate_profile["NumberOfTweets"] / legitimate_profile["AccountLifetime"]
    )

    # Read legitimate users followings data
    legitimate_followings = pd.read_table(
        "Datasets/legitimate_users_followings.txt",
        header=None,
        names=["UserID", "SeriesOfNumberOfFollowings"],
    )

    # Read legitimate users tweets data
    legitimate_tweets = pd.read_table(
        "Datasets/legitimate_users_tweets.txt",
        header=None,
        names=["UserID", "TweetID", "Tweet", "CreatedAt"],
    )

    # Return all dataframes
    return {
        "polluters_profile": polluters_profile,
        "polluters_followings": polluters_followings,
        "polluters_tweets": polluters_tweets,
        "legitimate_profile": legitimate_profile,
        "legitimate_followings": legitimate_followings,
        "legitimate_tweets": legitimate_tweets,
    }


# Reading all the files and preprocess some of them.
data = read_and_process_files()
# data["polluters_profile"].head()


# Separation de data
df_polluters = data["polluters_profile"]
df_polluters_tweets = data["polluters_tweets"]
df_polluters_followings = data["polluters_followings"]

# ===============================Legitimate data=========================================

df_legitimate = data["legitimate_profile"]
df_legitimate_tweets = data["legitimate_tweets"]
df_legitimate_followings = data["legitimate_followings"]


# ====================================== Creating the attribut restants=========================
# Fonction pour compter les URLs dans un tweet
def count_urls(text):
    urls = re.findall(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        str(text),
    )
    return len(urls)


# Compter le nombre d'URL dans chaque tweet
df_polluters_tweets["URL_Count"] = df_polluters_tweets["Tweet"].apply(count_urls)
df_polluters_tweets["Proportion_URL"] = df_polluters_tweets[
    "URL_Count"
] / df_polluters_tweets["Tweet"].notna().astype(int)

# Apply URL count function to legitimate tweets
df_legitimate_tweets["URL_Count"] = df_legitimate_tweets["Tweet"].apply(count_urls)
df_legitimate_tweets["Proportion_URL"] = df_legitimate_tweets[
    "URL_Count"
] / df_legitimate_tweets["Tweet"].notna().astype(int)


# time difference between tweets
# Calcul du Temps moyen et maximal entre deux tweets consécutifs


def calcul_time_diff(df):
    # Convertir la colonne CreatedAt en datetime
    df["CreatedAt"] = pd.to_datetime(df["CreatedAt"])

    # Trier les tweets par utilisateur et par date
    df = df.sort_values(by=["UserID", "CreatedAt"])

    # Calculer la différence de temps entre deux tweets consécutifs
    df["Time_Diff"] = df.groupby("UserID")["CreatedAt"].diff()

    # Convertir la différence de temps en secondes
    df["Time_Diff_Seconds"] = df["Time_Diff"].dt.total_seconds()

    # Calculer le temps moyen et maximal entre deux tweets consécutifs par utilisateur
    time_stats = (
        df.groupby("UserID")["Time_Diff_Seconds"].agg(["mean", "max"]).reset_index()
    )
    time_stats.columns = [
        "UserID",
        "Mean_Time_Between_Tweets",
        "Max_Time_Between_Tweets",
    ]
    return time_stats


def calculate_at_proportion(df):
    """
    Calculate the proportion of '@' mentions in the 'Tweet' column of a dataframe.

    Parameters:
        df (pd.DataFrame): The input dataframe containing a 'Tweet' column.

    Returns:
        pd.DataFrame: The modified dataframe with new columns:
                     - 'count_AT': Number of '@' mentions in each tweet.
                     - 'Tweet_Length': Length of each tweet.
                     - 'Proportion_AT': Proportion of '@' mentions in each tweet.
    """
    # Ensure the 'Tweet' column is treated as a string
    df["Tweet"] = df["Tweet"].astype(str)

    # Count the number of '@' mentions in each tweet
    df["count_AT"] = df["Tweet"].apply(lambda x: x.count("@"))

    # Calculate the length of each tweet
    df["Tweet_Length"] = df["Tweet"].apply(len)

    # Calculate the proportion of '@' mentions
    df["Proportion_AT"] = df.apply(
        lambda row: (
            row["count_AT"] / row["Tweet_Length"] if row["Tweet_Length"] != 0 else 0
        ),
        axis=1,
    )

    return df


# ==============================porportion at ==================================
df_legitimate_tweets = calculate_at_proportion(df_legitimate_tweets)
df_polluters_tweets = calculate_at_proportion(df_polluters_tweets)

# Pour les donnees polluters
time_diff_polluters = calcul_time_diff(df_polluters_tweets)

df_polluters = df_polluters.merge(time_diff_polluters, on="UserID", how="left")
# Pour les donnees Legitimate
time_diff_legitimate = calcul_time_diff(df_legitimate_tweets)

df_legitimate = df_legitimate.merge(time_diff_legitimate, on="UserID", how="left")


# pour les donnees legitimate et polluters on va selectionner les colonnes qu'on veut seulement ici

df_no_duplicates_polluters_tweets = df_polluters_tweets.drop_duplicates(
    subset="UserID", keep="first"
)

df_no_duplicates_legitimate_tweets = df_legitimate_tweets.drop_duplicates(
    subset="UserID", keep="first"
)


df_polluters_tweets = df_no_duplicates_polluters_tweets[
    ["URL_Count", "Proportion_URL", "count_AT", "Tweet_Length", "Proportion_AT"]
]
df_legitimate_tweets = df_no_duplicates_legitimate_tweets[
    ["URL_Count", "Proportion_URL", "count_AT", "Tweet_Length", "Proportion_AT"]
]

# Reinitialisation des indexes pour pouvoir combinner les donnes
df_legitimate_tweets = df_legitimate_tweets.reset_index(drop=True)
df_legitimate = df_legitimate.reset_index(drop=True)

df_polluters = df_polluters.reset_index(drop=True)
df_polluters_tweets = df_polluters_tweets.reset_index(drop=True)

# Now we are gonna delete some features that we don't need on both files legitimate and polluters
df_polluters = df_polluters.drop(["UserID", "CreatedAt", "CollectedAt"], axis=1)
df_legitimate = df_legitimate.drop(["UserID", "CreatedAt", "CollectedAt"], axis=1)

#  Now we are gonna combine the file based on polluters and legitimates
df_polluters_final = pd.concat([df_polluters, df_polluters_tweets], axis=1)

df_legitimate_final = pd.concat([df_legitimate, df_legitimate_tweets], axis=1)

# Add a 'Class' column to indicate polluters (1) and legitimate users (0)
df_polluters_final["Class"] = 1  # Polluters
df_legitimate_final["Class"] = 0  # Legitimate users

# Combining the two datasets into a single one and save the result on a csv format
df_final = pd.concat(
    [df_polluters_final, df_legitimate_final], axis=0, ignore_index=True
)
df_final.to_csv("Combined_datasets.csv", ignore_index=True)
