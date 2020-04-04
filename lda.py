import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

text_path = 'data/preprocessed_text.json'
with open('data/preprocessed_text.json', 'r') as f:
    data = json.load(f)
all_text = ' '.join([paper_json['body'] for paper_json in data.values()])

wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

wordcloud.generate(all_text[:])

# wordcloud.to_image()
plt.imshow(wordcloud)
plt.show()

