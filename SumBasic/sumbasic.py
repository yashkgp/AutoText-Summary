import nltk, sys, glob
from imp import reload
reload(sys)  

lemmatize = True
rm_stopwords = True
num_sentences = int(sys.argv[1])
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()

def clean_sentence(tokens):
	tokens = [t.lower() for t in tokens]
	if lemmatize: tokens = [lemmatizer.lemmatize(t) for t in tokens]
	if rm_stopwords: tokens = [t for t in tokens if t not in stopwords]
	return tokens

def get_probabilities(cluster, lemmatize, rm_stopwords):
	word_ps = {}
	token_count = 0.0
	for path in cluster:
		with open(path) as f:
			tokens = clean_sentence(nltk.word_tokenize(f.read()))
			token_count += len(tokens)
			for token in tokens:
				if token not in word_ps:
					word_ps[token] = 1.0
				else:
					word_ps[token] += 1.0
	for word_p in word_ps:
		word_ps[word_p] = word_ps[word_p]/float(token_count)
	return word_ps

def get_sentences(cluster):
	sentences = []
	for path in cluster:
		with open(path) as f:
			sentences = f.readlines()
	return sentences

def clean_sentence(tokens):
	tokens = [t.lower() for t in tokens]
	if lemmatize: tokens = [lemmatizer.lemmatize(t) for t in tokens]
	if rm_stopwords: tokens = [t for t in tokens if t not in stopwords]
	return tokens

def score_sentence(sentence, word_ps):
	score = 0.0
	num_tokens = 0.0
	sentence = nltk.word_tokenize(sentence)
	tokens = clean_sentence(sentence)
	for token in tokens:
		if token in word_ps:
			score += word_ps[token]
			num_tokens += 1.0
	return float(score)/float(num_tokens)

def max_sentence(sentences, word_ps, simplified):
	max_sentence = None
	max_score = 0
	for sentence in sentences:
		score = score_sentence(sentence, word_ps)
		if score > max_score or max_score == 0:
			max_sentence = sentence
			max_score = score
	if not simplified: update_ps(max_sentence, word_ps)
	return max_sentence

def update_ps(max_sentence, word_ps):
	sentence = nltk.word_tokenize(max_sentence)
	sentence = clean_sentence(sentence)
	for word in sentence:
		word_ps[word] = word_ps[word]**2
	return True

def orig(cluster):
	cluster = glob.glob(cluster)
	word_ps = get_probabilities(cluster, lemmatize, rm_stopwords)
	sentences = get_sentences(cluster)
	summary = []
	for i in range(num_sentences):
		summary.append(max_sentence(sentences, word_ps, False))
	return "".join(summary)

def simplified(cluster):
	cluster = glob.glob(cluster)
	word_ps = get_probabilities(cluster, lemmatize, rm_stopwords)
	sentences = get_sentences(cluster)
	summary = []
	for i in range(num_sentences):
		summary.append(max_sentence(sentences, word_ps, True))
	return "".join(summary)

def leading(cluster):
	cluster = glob.glob(cluster)
	sentences = get_sentences(cluster)
	summary = []
	for i in range(num_sentences):
		summary.append(sentences[i])
	return "".join(summary)

def main():
	cluster = sys.argv[2]
	summary = eval("orig" + "('" + cluster + "')")
	a = open(sys.argv[3],"w")
	for line in summary:
		a.write(line)
	a.close()

if __name__ == '__main__':
	main()
