import numpy
import nltk
nltk.download('stopwords')
from collections import Counter
from itertools import combinations
from typing import List
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import math
from flask import Flask, render_template, request
from NLP import TextSummarizer
from funny_image import WordCloudGenerator

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        shortness = request.form['input1']
        input_text = request.form['input2']
        with open('main_page.txt', 'w') as f:
            f.write(input_text)
        my_class = TextSummarizer()
        w = WordCloudGenerator()
        w.generate_wordcloud('static/wordcloud.png')
        output_text = my_class.summary(int(shortness),'main_page.txt')[0]
        num_of_sents = my_class.summary(int(shortness),'main_page.txt')[1]
        result = f"Summary at shortness level {shortness} with a total of {num_of_sents} sentences:  {output_text}"
        return render_template('main_page.html', output=result,image_path='wordcloud.png')
    else:
        return render_template('main_page.html')

@app.route('/tech_stack')
def tech_stack():
    return render_template('tech_stack.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')

if __name__ == '__main__':
    app.run()