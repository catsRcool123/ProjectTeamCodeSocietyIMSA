from NLP import TextSummarizer
import nltk

nltk.download('stopwords')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
import string
import matplotlib

matplotlib.use('Agg')


class WordCloudGenerator:

    def __init__(self):
        self.ts = TextSummarizer()
        self.stop_words = set(stopwords.words("english"))
        self.translator = str.maketrans("", "", string.punctuation)

    def generate_wordcloud(self, filename):
        sentences = self.ts.read("main_page.txt")
        words = [w.lower().translate(self.translator) for sentence in sentences for w in sentence.split() if
                 w.lower() not in self.stop_words]
        word_freq = Counter(words)
        wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
