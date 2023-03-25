import numpy as np
from flask import Flask, Response, request, render_template
import pickle
import pandas as pd
from for_flask import recomendation

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/handle_data', methods=['POST'])
def handle_data():
    idx = request.form.get('article_number')
    no_of_news_article = request.form.get('articles_amount')
    rec = recomendation(idx = int(idx), no_of_news_article = int(no_of_news_article))

    return Response(
       rec.to_csv(),
       mimetype="text/csv",
       headers={"Content-disposition":
       "attachment; filename=filename.csv"})

if __name__ == "__main__":
    app.run(debug=True)
