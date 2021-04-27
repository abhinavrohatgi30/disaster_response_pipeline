import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the message data and categories data into a single dataframe
    :param messages_filepath: Message data file path
    :param categories_filepath: Categories data file path
    :return: Merged dataframe
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge the above datasets on the id field
    df = messages.merge(categories, on="id")
    return df


def clean_data(df):
    """
    The raw data in the dataframe is cleaned, structured, de-duplicated and processed
    :param df: The dataframe containing the raw data
    :return: cleaned dataframe
    """

    categories = df.categories.str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.head(1)

    category_colnames = [value[0:-2] for value in row.values[0]]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: None if int(x) > 1 else int(x))

    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # dropping rows having null values in any of the category columns
    df = df.dropna(subset=category_colnames, how="any")

    # Remove duplicate data
    df = df.drop_duplicates(subset='message', keep="last")
    return df


def save_data(df, database_filename):
    """
    Saves the data into a database file on the file system
    :param df: Dataframe to save
    :param database_filename: File path of the processed dataframe
    :return: None
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterMessages', engine, index=False)


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
