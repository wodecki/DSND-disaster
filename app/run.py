import json
import plotly
import pandas as pd
import numpy as np
import sklearn

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/disaster.db')
df = pd.read_sql_table('DisastersData', engine)

# load model
model = joblib.load("../model/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


   # --------  Fig 1. Created by Udacity  --------
    figures = []
    fig1 = {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }

    # --------  Fig 2. Distribution of Message Categories  --------
    categories = df[df.columns[4:]]
    sums = categories.sum()
    fig2 = {
        'data': [
            Bar(
                x=sums.index,
                y=sums.values
            )
        ],

        'layout': {
            'title': 'Distribution of Message Categories',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Category"
            }
        }
    }

    # --------  Fig 3. Distribution of Message Categories  --------
    
    sums.nlargest(5).index
    top_5_cats = df[sums.nlargest(5).index].columns
    top_5s = list(df[sums.nlargest(5).index].columns)
    top_5s.extend(['genre', 'message'])

    top5 = df[top_5s]
    genre_counts_top5 = top5.groupby('genre').sum()

    fig3 = {
        'data': [
            Bar(name=top_5_cats[0], x=genre_names, y=genre_counts_top5[top_5_cats[0]]),
            Bar(name=top_5_cats[1], x=genre_names, y=genre_counts_top5[top_5_cats[1]]),
            Bar(name=top_5_cats[2], x=genre_names, y=genre_counts_top5[top_5_cats[2]]),
            Bar(name=top_5_cats[3], x=genre_names, y=genre_counts_top5[top_5_cats[3]]),
            Bar(name=top_5_cats[4], x=genre_names, y=genre_counts_top5[top_5_cats[4]])
        ],

        'layout': {
            'title': 'Distribution of Message Genres for TOP 5 Categories',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Genre"
            },
            'barmode': 'stack'
        }
    }

    # --------  Create a list of figures to render  --------
    figures.append(fig1)
    figures.append(fig2)
    figures.append(fig3)

    # plot ids for the html id figures
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # encode plotly graphs in JSON
    ids = ["figure-{}".format(i) for i, _ in enumerate(figures)]
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly figures
    return render_template('master.html', ids=ids, figuresJSON=figuresJSON, data_set=df)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
