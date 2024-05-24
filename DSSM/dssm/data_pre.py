import collections
import pandas as pd
import numpy as np


user = pd.read_csv('../ml-1m/users.dat',sep='::',header=None,engine='python',encoding='latin1')
movie = pd.read_csv('../ml-1m/movies.dat',sep='::',header=None,engine='python',encoding='latin1')
rating = pd.read_csv('../ml-1m/ratings.dat',sep='::',header=None,engine='python')

user.columns = ["user_id","gender","age","occupation","zip_code"]
movie.columns = ["movie_id","movie_name","genre"]
rating.columns = ["user_id","movie_id","score","time_print"]

genre_count = collections.defaultdict(int)
for genres in movie["genre"].str.split("|"):
    for genre in genre_count:
        genre_count[genre]+=1
def count_max_genre(x):
    sub_count={}
    for y in x.split("|"):
        sub_count[y]=genre_count[y]

    return sorted(sub_count.items(), key=lambda x: x[1], reverse=True)[0][0]
movie["genre"]=movie["genre"].map(count_max_genre)

data_summary = pd.merge(pd.merge(user,rating,on="user_id"),movie,on="movie_id")
new_colums = ["user_id","gender","age","occupation","movie_id","genre",
               "score","movie_name","zip_code","time_print"]
data_pre = data_summary.reindex(columns=new_colums)

gender_map = {
    "F":0 ,
    "M":1
}

age_map = {
    1:0,
    18:1,
    25: 2,
    35: 3,
    45: 4,
    50: 5,
    56: 6
}
data_pre["gender"] = data_pre["gender"].map(gender_map)
data_pre["age"] = data_pre["age"].map(age_map)
data_pre["genre"], genre_labels = pd.factorize(data_pre["genre"])

data_pre.iloc[:,:7].to_csv("data_pre.txt",index=False,header=True)

print(data_pre.shape)
print(data_pre.head())
print(data_pre.iloc[0])


