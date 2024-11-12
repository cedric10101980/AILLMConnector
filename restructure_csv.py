import pandas as pd
import os
import glob
import random

# Get all files in the directory
files = glob.glob('docs/Call Records/Dial*')

for file in files:
    # Get the base name of the file
    base = os.path.basename(file)

     # Add '_modified' to the base name
    new_base = os.path.splitext(base)[0] + '_modified' + os.path.splitext(base)[1]

    # Read the CSV file
    df = pd.read_csv(file)

    # Convert the 'TimeStamp' column to datetime
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format='%d-%m-%Y %H:%M')

    # Sort the dataframe by 'TimeStamp' in ascending order
    df = df.sort_values(['PhoneNumber','TimeStamp'])

    # Convert the 'TimeStamp' column to datetime
    #df['TimeStamp'] = pd.to_datetime(df['TimeStamp'],format='%d-%m-%Y %H:%M')

    # Replace the month with a random number between 1 and 4
    #df['TimeStamp'] = df['TimeStamp'].apply(lambda x: x.replace(month=random.randint(1, 4), day=random.randint(1, 20)))

    # Convert the 'TimeStamp' column to DD/MM/YYYY format
    df['TimeStamp'] = df['TimeStamp'].dt.strftime('%d/%m/%Y %H:%M')

    # Write the DataFrame to a new CSV file
    df.to_csv(new_base, index=False)