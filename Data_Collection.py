from Google_Search_Engine import search_linkedin_profiles, api_keys, cse_ids
from random_forest import get_dataset_path, has_content
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import time
import random

def preprocess_dataframe(df):
    string_columns = [
        'Full Name', 'Workplace', 'Location', 'Connections', 'Photo',
        'Followers', 'About', 'Experiences', 'Educations', 'Licenses', 
        'Volunteering', 'Skills', 'Recommendations', 'Projects', 'Publications',
        'Courses', 'Honors', 'Scores', 'Languages', 'Organizations', 'Interests', 'Activities'
    ]
    df[string_columns] = df[string_columns].fillna('')
    
    df['Has_Photo'] = df['Photo']
    df['Has_About'] = df['About'].apply(has_content)
    df['Has_Experience'] = df['Experiences'].apply(has_content)
    df['About_Length'] = df['About'].apply(len)
    
    #add an ID column
    df['ID'] = df.index

    count_fields = [
        'Number of Experiences', 'Number of Educations', 'Number of Licenses',
        'Number of Volunteering', 'Number of Skills', 'Number of Recommendations',
        'Number of Projects', 'Number of Publications', 'Number of Courses',
        'Number of Honors', 'Number of Scores', 'Number of Languages',
        'Number of Organizations', 'Number of Interests', 'Number of Activities',
        'Connections', 'Followers'
    ]
    df[count_fields] = df[count_fields].fillna(0)
    return df

def main():
    index_creds = 0
    path = "/home/fasma/heriot-watt/Research_Methods/FADL/datasets/trainning_Datasets/LinkedIn people profiles datasets - Clean_label_data_NoAI.csv"
    df = pd.read_csv(path)
    df = preprocess_dataframe(df)

    if 'Label' not in df.columns:
        raise ValueError("The dataset must contain a 'Label' column with 0 or 1 values.")
    print("Size of the dataset:", df.shape)
    search_queries = {"Name" : [], "Workplace": [], "Location": [], "ID": []}

    # start from a specific row
    start_row = 2100 #last row was 741 on june 1 2025
    end_row = 2500 # Adjust this to the desired end row
    df = df.iloc[start_row:end_row]
    print(f"Processing dataset from row {start_row} to {end_row}. Total rows: {df.shape[0]}")

    df = df[df['Has_Photo'] == True]
    print(f"Filtered dataset size: {df.shape[0]} rows with photos")
    
    for _, row in df.iterrows():
        search_queries["Name"].append(row['Full Name'])
        search_queries["Workplace"].append(row['Workplace'])
        search_queries["Location"].append(row['Location'])
        search_queries['ID'].append(row['ID'])

    print("Search queries size:", len(search_queries["Workplace"]))
    
    for _, name in enumerate(search_queries["Name"]):
        query = f'site:linkedin.com/in/ "{name}" "{search_queries["Workplace"][_]}" "{search_queries["Location"][_]}"'
        print(f"Searching for: {query}")
        ret = search_linkedin_profiles(query, api_keys[index_creds], cse_ids[index_creds], search_queries['ID'][_], image_download=True, 
                                pp_name=f"datasets/profile_picture_data/profile_pictures/linkedin_profile_picture_{search_queries['ID'][_]}.jpg", 
                                output_json=f"datasets/profile_picture_data/url.json")
        if ret == 1:
            print(f"Error occurred while searching for {name} at this index {search_queries['ID'][_]}. Retrying with next API key.")
            index_creds += 1
        if index_creds >= len(api_keys):
            print("All API keys exhausted. Stopping the search.")
            return
    print("Search completed.")
main()