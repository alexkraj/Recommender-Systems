# CTVT58
# CC - Recommender Systems
# Heavily based off Web Tech assignment which uses matrix factorisation
from flask import Flask, render_template, request
from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np
import random
import json

app = Flask(__name__)
user = 6
user_songs = []  # stores song IDs that the user has already rated

songs_df = []
ratings_df = []
context = 0
duration = 0
num_songs = [8, 16, 24, 30]
playlist_length = 0


# read all the SONGS and RATINGS and convert into panda dataframe
def read_data(d_songs, d_ratings):

    data_songs = [i.strip().split(",") for i in open(d_songs, "r").readlines()]
    songs_df = pd.DataFrame(
        data_songs, columns=["Song_ID", "Artist", "Title", "Genre"]
    )
    songs_df["Song_ID"] = songs_df["Song_ID"].apply(pd.to_numeric, errors="coerce")

    data_ratings = [i.strip().split(",") for i in open(d_ratings, "r").readlines()]
    ratings_df = pd.DataFrame(
        data_ratings, columns=["User_ID", "Song_ID", "Rating", "Landscape"], dtype=int
    )

    ratings_df["User_ID"] = ratings_df["User_ID"].apply(pd.to_numeric, errors="coerce")
    ratings_df["Song_ID"] = ratings_df["Song_ID"].apply(pd.to_numeric, errors="coerce")
    ratings_df["Rating"] = ratings_df["Rating"].apply(pd.to_numeric, errors="coerce")
    ratings_df["Landscape"] = ratings_df["Landscape"].apply(pd.to_numeric, errors="coerce")
    return songs_df, ratings_df

# reads the .csv files into the ratings and songs dataframes respectively
# drops the landscape column as all of the contexts have been filtered
def filter_df(landscape_value):
    global ratings_df, songs_df
    songs_df, ratings_df = read_data("data/Pre-Filtered/music_data.csv", "data/Pre-Filtered/ratings_data.csv")
    # remove all the ratings not made in the same context
    ratings_df = ratings_df[ratings_df.Landscape == landscape_value]
    # drop the context column because pre-filtering is now complete
    ratings_df = ratings_df.drop('Landscape', 1)


# making the recommendations matrix, fill the rest with 0s
def create_matrix(ratings):
    recommendation_matrix = ratings.pivot(
        index="User_ID", columns="Song_ID", values="Rating"
    ).fillna(0)
    return recommendation_matrix


# demeaning the data
def demean_data(ratings):
    R = ratings.rename_axis("ID").values
    user_ratings_mean = np.mean(R, axis=1)
    demeaned_ratings = R - user_ratings_mean.reshape(-1, 1)
    return demeaned_ratings, user_ratings_mean


def recommend_songs(predictions_df, userID, songs_df, original_ratings_df, num_recommendations=50):
    user_row_number = userID
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(
        ascending=False
    )

    user_data = original_ratings_df[original_ratings_df.User_ID == (userID)]
    user_full = user_data.merge(
        songs_df, how="left", left_on="Song_ID", right_on="Song_ID"
    ).sort_values(["Rating"], ascending=False)
    # print("User {0} has already rated {1} books.".format(userID, user_full.shape[0]))
    print(
        "Recommending the highest {0} predicted new songs for the listener in their given context.".format(
            num_recommendations
        )
    )

    recommendations = (
        songs_df[~songs_df["Song_ID"].isin(user_full["Song_ID"])]
        .merge(
            pd.DataFrame(sorted_user_predictions).reset_index(),
            how="left",
            left_on="Song_ID",
            right_on="Song_ID",
        )
        .rename(columns={user_row_number: "Predictions"})
        .sort_values("Predictions", ascending=False)
        .iloc[:num_recommendations, :-1]
    )
    return user_full, recommendations

############################### FLASK METHODS BELOW ###############################

@app.route("/")
def home():
    message = ""
    user_info = {"user_ID": user, "message": message}
    return render_template("home.html", user=user_info)

@app.route("/getContext", methods=['POST'])
def getContext():
    global context
    context = int(request.form["context"])
    filter_df(context)
    return "successfully filtered the data by context" 

@app.route("/getDuration", methods=['POST'])
def getDuration():
    global playlist_length
    duration = int(request.form["duration"])
    playlist_length = num_songs[duration]
    return "successfully determined playlist length to match drive length" 


@app.route("/myrecc", methods=["GET"])
def my_recc():
    # create the matrix
    global ratings_df, playlist_length
    R_df = create_matrix(ratings_df)

    # demean the data
    R_demeaned, user_ratings_mean = demean_data(R_df)

    # singular value decomposition
    U, sigma, Vt = svds(R_demeaned, k=20)
    sigma = np.diag(sigma)

    # making predictions from decomposed matrices
    all_user_predicted_ratings = np.dot(
        np.dot(U, sigma), Vt
    ) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=R_df.columns)

    already_rated, predictions = recommend_songs(
        preds_df, user, songs_df, ratings_df, playlist_length
    )

    recommended_songs = predictions.head(playlist_length)
    # recommended_songs = already_rated.head(playlist_length)
    songs_json = recommended_songs.to_json(orient="records")
    # print(songs_json)
    user_info = {"user_ID": user, "rec_songs": songs_json}

    return json.dumps(user_info)


if __name__ == "__main__":
    app.run(debug=True)
