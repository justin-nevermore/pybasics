from sklearn.feature_extraction.text import TfidfVectorizer

s1="The car is driven on the road.".replace(".","")
s2="The truck is driven on the highway.".replace(".", "")
vectorizer= TfidfVectorizer()
response=vectorizer.fit_transform([s1, s2])


print(vectorizer.get_feature_names())
print(response.toarray())