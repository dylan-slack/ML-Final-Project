import re
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.model_selection import train_test_split

def get_IMBD_preprocessing(load_cached=True):
	"""
	Loads IMBD preprocessing according to Pouya's preprocessing.  Gives access to caching functionality to make the 
	process a bit quicker.

	Params:
	---------
	load_cached : bool

	Returns:
	---------
	IMBD train, test, val splits
	"""

	if not load_cached:
		"""
		IMBD preprocessing taken from:
			- https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184
		"""

		reviews_train = []
		for line in open('aclImdb//movie_data/full_train.txt', 'r'):
			reviews_train.append(line.strip())
		reviews_test = []

		for line in open('aclImdb//movie_data/full_test.txt', 'r'):
			reviews_test.append(line.strip())

		REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
		REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

		def preprocess_reviews(reviews):
			reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
			reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
			return reviews

		reviews_train_clean = preprocess_reviews(reviews_train)
		reviews_test_clean = preprocess_reviews(reviews_test)

		cv = CountVectorizer(binary=True)
		cv.fit(reviews_train_clean)
		X = cv.transform(reviews_train_clean)
		X_test = cv.transform(reviews_test_clean)

		target = [1 if i < 12500 else 0 for i in range(25000)]
		X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

		data = {"X_train": X_train, "X_val": X_val, "X_test": X_test, "y_train":y_train, "y_val": y_val, "y_test":target}
		with open('data/data_cache.pickle', 'wb') as handle:
			pickle.dump(data, handle)
	else:
		with open('data/data_cache.pickle', 'rb') as handle:
			data = pickle.load(handle)

	return data


