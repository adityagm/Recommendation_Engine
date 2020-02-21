import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

movies_df = pd.read_csv('movies.csv')

''' since the title field is title+(year), we need to split the string so that the movies are identified by the title properly'''
""" extract the year and add to the year column"""
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)
''' extract the year without the parentheses and replace the old values with the new ones'''
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)
''' replace the year part of the title string with whitespaces '''
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')

''' remove the whitespace that we added to the title string earlier using the strip function '''
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

movies_df['genres'] = movies_df.genres.str.split('|')

movies_df_copy = movies_df.copy()

''' creating one-hot array 1-true 0-false '''
for index, rows in movies_df.iterrows():
    for genre in rows['genres']:
        movies_df_copy.at[index, genre] = 1

''' filling the NaN values with 0'''

movies_df_copy = movies_df_copy.fillna(0)

#print(movies_df_copy.head())

movies_df.to_csv(path_or_buf='movies_edited.csv')

'''loading the ratings dataframe and dropping the column that is unnecessary'''
# ratings_df = pd.read_csv('ratings.csv')
# ratings_df = ratings_df.drop('timestamp', 1)
# print(ratings_df.head(5))

'''rating data of movies from the user corresponding to the title of the movies'''
userInput = [
    {'title': 'Breakfast Club, The', 'rating': 5},
    {'title': 'Toy Story', 'rating': 3.5},
    {'title': 'Jumanji', 'rating': 2},
    {'title': "Pulp Fiction", 'rating': 5},
    {'title': 'Akira', 'rating': 4.5}
]
userInput_df = pd.DataFrame(userInput)

'''since the userInput_df is not yet complete since the ID of the movie is missing'''
movieID_df = movies_df[movies_df['title'].isin(userInput_df['title'].tolist())]
print(movieID_df)

inputMovies = pd.merge(movieID_df, userInput_df)

inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
print(inputMovies)

userMovies = movies_df_copy[movies_df_copy['movieId'].isin(inputMovies['movieId'].tolist())]
userMovies = userMovies.reset_index(drop=True)
print(userMovies)

'''aggregating the one-hot array dataframe'''
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
print(userGenreTable)

print(inputMovies['rating'])

'''creating the user profile i.e. the weighted array'''
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
print(userProfile)

genreTable = movies_df_copy.set_index(movies_df_copy['movieId'])
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
print(genreTable)
# genreTable.shape - will generate the dimensions of the dataframe

# multiply the genres with the weights of the genres calculated previously then take the weighted average
recommendation_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())

recommendation_df = recommendation_df.sort_values(ascending=False)
print(recommendation_df.head())

# generating the recommendation table
print(movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(20).keys())])