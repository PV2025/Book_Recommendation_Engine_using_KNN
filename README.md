# Book_Recommendation_Engine_using_KNN

get_recommends("Some Book Title")


# Count ratings per user
user_counts = ratings['user_id'].value_counts()
ratings = ratings[ratings['user_id'].isin(user_counts[user_counts >= 200].index)]

# Count ratings per book
book_counts = ratings['book_id'].value_counts()
ratings = ratings[ratings['book_id'].isin(book_counts[book_counts >= 100].index)]


ratings_matrix = ratings.pivot_table(index='book_id', columns='user_id', values='rating').fillna(0)


from sklearn.neighbors import NearestNeighbors

# Fit model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(ratings_matrix.values)


book_titles = books[['book_id', 'title']]
book_title_to_id = dict(zip(book_titles['title'], book_titles['book_id']))
book_id_to_title = dict(zip(book_titles['book_id'], book_titles['title']))


def get_recommends(book_title):
    book_id = book_title_to_id.get(book_title)

    if book_id is None:
        return [book_title, []]

    book_index = ratings_matrix.index.tolist().index(book_id)
    distances, indices = model.kneighbors([ratings_matrix.iloc[book_index]], n_neighbors=6)

    recommends = []
    for i in range(1, len(distances[0])):  # Skip the first one (it's the book itself)
        similar_book_id = ratings_matrix.index[indices[0][i]]
        recommends.append([book_id_to_title[similar_book_id], distances[0][i]])

    return [book_title, recommends]


get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")


[
  'The Queen of the Damned (Vampire Chronicles (Paperback))',
  [
    ['Catch 22', 0.793983519077301], 
    ['The Witching Hour (Lives of the Mayfair Witches)', 0.7448656558990479], 
    ['Interview with the Vampire', 0.7345068454742432],
    ['The Tale of the Body Thief (Vampire Chronicles (Paperback))', 0.5376338362693787],
    ['The Vampire Lestat (Vampire Chronicles, Book II)', 0.5178412199020386]
  ]
]


