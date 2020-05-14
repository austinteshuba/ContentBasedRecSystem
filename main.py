#Dataframe manipulation library
import pandas as pd
#Math functions, we'll only need the sqrt function so let's import only that
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Get the data

movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")

# These are some printing options so we can see all of the data

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# See the data
print(movies_df.head(10))
print(ratings_df.head(10))

# Step 2: Preprocessing

# Since the year is also a feature that could be used for a recommendation
# we should seperate it

movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

print(movies_df.head())

# Now, we need to handle the genres.
# THe genres should be held in a listof genres for easier access, rather than seperated by |

movies_df['genres'] = movies_df.genres.str.split('|')
print(movies_df.head())

# THis is also pretty inefficient. We should use one-hot encoding
moviesWithGenres_df = movies_df.copy()

#For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
print(moviesWithGenres_df.head())

# For the ratings DF, we can drop the timestamp entry.
ratings_df.drop("timestamp", axis=1, inplace=True)
print(ratings_df.head())

# We can simulate a user input
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ]
inputMovies = pd.DataFrame(userInput)

# We can now match these movies to their movie ID from the moviesdf

#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

inputMovies = pd.merge(inputId, inputMovies)
# Get rid of the genres and year
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
print(inputMovies.head())

#Get the user movies with genre data
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
#Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop=True)
#Dropping unnecessary issues due to save memory and to avoid issues
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
print(userGenreTable.head())
print(inputMovies["rating"])

# Make the user profile, now that we have the movies matrix and the ratings matrix


# Step 3: Recommendation systems. We need the user profile
#Dot product to get weights
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])

print(userProfile.head())

#Now let's get the genres of every movie in our original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
#And drop the unnecessary information
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
print(genreTable.head())

# Now, we can make a weighted candidate matrix by multiplying the user profile with the movies matrix
#Multiply the genres by the weights and then take the weighted average
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
print(recommendationTable_df.head())


#Sort our recommendations in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
#Just a peek at the values
print(recommendationTable_df.head())

# Cool. Now we can print the result
print(movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())])
