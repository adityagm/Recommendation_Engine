import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

# import the movies csv as a dataframe
movies_df = pd.read_csv('movies.csv')

# import the ratings csv as a dataframe
ratings_df = pd.read_csv('ratings.csv')

# drop the timestamp in the ratings csv, since we dont need it
ratings_df = ratings_df.drop('timestamp', 1)
print(ratings_df.head())

# pandas has an extration function that can be used to copy a specific part of string using regexs
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)

# removing the year from the title field
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

movies_df['genres'] = movies_df.genres.str.split('|')

movies_df_copy = movies_df.copy()

''' creating one-hot array 1-true 0-false '''
for index, rows in movies_df.iterrows():
    for genre in rows['genres']:
        movies_df_copy.at[index, genre] = 1

''' filling the NaN values with 0'''
movies_df_copy = movies_df_copy.fillna(0)
print(movies_df.head())

'''rating data of movies from the user corresponding to the title of the movies'''
userInput = [
    {'title': 'Breakfast Club, The', 'rating': 5},
    {'title': 'Toy Story', 'rating': 3.5},
    {'title': 'Jumanji', 'rating': 2},
    {'title': "Pulp Fiction", 'rating': 5},
    {'title': 'Akira', 'rating': 4.5}
]
userInput = pd.DataFrame(userInput)

# we don't need the genres column  since collaborative recommendation engine depends on the rating and not the genres
movies_df_copy = movies_df_copy.drop('genres', 1)
movies_df_copy.to_csv(path_or_buf='movies_edited.csv')

similar_movies = movies_df_copy[movies_df_copy['title'].isin(userInput['title'].tolist())]
print(similar_movies.head())

inputID_df = movies_df[movies_df['title'].isin(userInput['title'].tolist())]
userInput = pd.merge(userInput, inputID_df)
print(userInput.head())

# check all the keys that are present in the userInput
print(userInput.keys())
userInput = userInput.drop('year', 1).drop('genres', 1)

userInput = userInput.set_index('movieId')
userInput = userInput.reset_index()
print(userInput.head())

# now we need the movie ratings of all the movies in the movie_df
userRatings = ratings_df[ratings_df['movieId'].isin(userInput['movieId'].tolist())]
print(userRatings.head())

# now we can create subgroups of the dataframes for each userId
userSubset = userRatings.groupby(['userId'])
print(userSubset.head())

print(userSubset.get_group(11325))
print("group keys", userSubset.groups.keys())
userSubsetGroup = sorted(userSubset, key=lambda x: len(x[1]), reverse=True)

# just take a small subset to study
userSubsetGroup = userSubsetGroup[0:100]

# creating a dictionary of pearson correlation where the keys are the userID and the values are the result of the pearson correlation coefficient
pearsonCorrelationCoef = {}

for name, group in userSubsetGroup:
    group = group.sort_values(by='movieId')
    userInput = userInput.sort_values(by='movieId')

    nRating = len(group)
    # will store all the movies that have been reviewed by both the user and the current group
    tempDf = userInput[userInput['movieId'].isin(group['movieId'].tolist())]
    # will store all the ratings of all the common movies reviewed by the group and the user
    tempRatingList = tempDf['rating'].tolist()
    # will store all the ratings of the group
    tempGroupRatingList = group['rating'].tolist()
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList), 2) / float(nRating)
    Syy = sum([j**2 for j in tempGroupRatingList]) - pow(sum(tempGroupRatingList), 2) / float(nRating)
    Sxy = sum([i*j for i,j in zip(tempRatingList, tempGroupRatingList)]) - sum(tempRatingList)*sum(tempGroupRatingList) / float(nRating)

    if Sxx != 0 and Syy != 0:
        pearsonCorrelationCoef[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationCoef[name] = 0

print(pearsonCorrelationCoef.items())

pearsonDf = pd.DataFrame.from_dict(pearsonCorrelationCoef, orient='index')
print(pearsonDf.head())
pearsonDf.columns = ['similarityIndex']
pearsonDf['userId'] = pearsonDf.index
pearsonDf.index = range(len(pearsonDf))
print(pearsonDf.head())

# get the top similar users profiles
topUsers = pearsonDf.sort_values(by='similarityIndex', ascending=False)[0:50]
print(topUsers.head())

# now to combine the ratings file and and the topUsers so that we can find the ratings of the users to all the movies
topUsersRatings = topUsers.merge(ratings_df, how="inner", left_on='userId', right_on='userId')
print(topUsersRatings.head())

# taking the weighted average of the ratings of the movies using the Pearson Correlation as the weight
# first multiply the similarity with the ratings
topUsersRatings['weightedSimilarity'] = topUsersRatings['similarityIndex']*topUsersRatings['rating']

# add all the weighted Similarity and the similarity of individual users across the movies
topUsersTemp = topUsersRatings.groupby('movieId').sum()[['weightedSimilarity', 'similarityIndex']]
topUsersTemp.columns = ['sum_weightedSimilarity', 'sum_similarityIndex']
# topUsersTemp = topUsersTemp.set_index('sum_similarityIndex')
# topUsersTemp = topUsersTemp.reset_index()
print(topUsersTemp.head())

# now take the weighted averages
# new dataframe to store the values
recommendation_df = pd.DataFrame()
recommendation_df['movieId'] = topUsersTemp.index
recommendation_df['weighted average recommendation score'] = topUsersTemp['sum_weightedSimilarity']/topUsersTemp['sum_similarityIndex']

# sort the values by the weighted average recommendation score in descending order
recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
print(recommendation_df.head(10))

# now print the names of the movies that have been recommended by the engine
# to locate the movie names in the original movie_df
print(movies_df.loc[movies_df['movieId'].isin(recommendation_df.head()['movieId'].tolist())])
