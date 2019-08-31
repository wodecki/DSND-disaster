import sys
import pandas as pd

import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Read source files with message text and their categories.

    Args:
    messages_filepath: a filepath to a messages CSV file
    categories_filepath: a filepath to a categories CSV file

    Returns:
    df: a pandas dataframe with categorized messages
    """
    messages =  pd.read_csv(messages_filepath)
    categories =  pd.read_csv(categories_filepath)

    # merge messages and categories df's by id
    df = pd.merge(messages, categories, on='id')

    # Split the values in the categories column on the ; character so that each value becomes a separate column
    categories = categories.categories.str.split(';', expand=True)

    # Convert category values to just numbers 0 or 1

    row = categories.iloc[0]
    categories.columns = row.apply(lambda x: x[:-2])

    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1] if int(x.split('-')[1]) < 2 else 1) #this line took me literally 2 full days, truly. There is some magic behind it's consequences for model evaluation...
        categories[column] = categories[column].astype(dtype='int32')

    # drop all but 'message' columns from `df`
    df.drop('categories', inplace=True, axis = 1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)

    return df

def clean_data(df):
    """
    Clean-up the dataframe with categorized messages: removed duplicates and NaNs

    Args:
    df: a pandas dataframe with categorized messages

    Returns:
    df: a cleaned pandas dataframe with merged messages and categories dataframes
    """
    categories = df.iloc[:, 4:]
    # drop duplicates
    df = df.drop_duplicates()

    # drop NaN's
    df = df.dropna(subset=categories.columns)
    # remove columns with only one category (there is no sense to train the model in such a case)
    cols_1_value = [c for c in categories.columns if (df[c].nunique() == 1)]
    df.drop(columns=cols_1_value, inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save the clean dataset into an sqlite database.

    Args:
    df: a pandas dataframe with categorized messages
    database_filename: a name for a database file, as a string ending with ".db" (e.g. 'my_database.db')
    Returns: None
    """
    database_table_name = database_filename[:-3]
    engine_name = 'sqlite:///'+database_filename

    engine = create_engine(engine_name)

    df.to_sql('DisastersData', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
