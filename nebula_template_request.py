import os

import pandas as pd
import requests
from dotenv import load_dotenv
from variables import *
import csvparser

load_dotenv('.env')

NEBULA_URL='https://nebula.cs.vu.nl/api/chat/completions'
NEBULA_TOKEN=str(os.getenv('NEBULA_TOKEN'))

def chat_with_model(model, system_prompt, user_prompt):
    url = NEBULA_URL
    headers = {
        'Authorization': f'Bearer {NEBULA_TOKEN}',
        'Content-Type': 'application/json',
    }
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                ]
            },
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()['choices'][0]['message']['content']

def classify_paper(file_df, system_prompt, user_prompt, model):
    # iterate through file_df and give prompt + abstract
    for index, row in file_df.iterrows():
        # if the classificatio has already been made, skip
        stored_generative_type = row['generative_type']

        if row["generative_type"] in GENERATIVE_TYPES:
            continue

        abstract = row['Abstract']
        title_and_subtitle = str(row['Title']) + " " + str(row["Subtitle"])
        user_prompt = USER_PROMPT_TEMPLATE.format(title=title_and_subtitle, abstract=abstract)

        if pd.isna(abstract) or abstract == '' or abstract == 'nan' or abstract == 'None' or len(abstract) < 5 or abstract == 'N/A':
            print(f"LLM response found for {row['UUID']}, skipping")
            continue

        llm_classification = chat_with_model(
            model=model,
            system_prompt=SYSTEM_PROMPT_TEMPLATE,
            user_prompt=user_prompt
        )

        # try catch
        try:
            # parse LLM input, e.g., response = "1,LLM", response = "0,none"
            classification = llm_classification.replace('"', "")
            classification = classification.split(",")
            is_generative = int(classification[0]) == 1
            generative_type = classification[1].strip()

            # append to dataframe
            file_df.at[index, 'is_generative'] = is_generative
            file_df.at[index, 'generative_type'] = generative_type
        except:
            print(f"{classification}, Error for {row['UUID']}, skipping")
            continue

        print(f"Classified {row['UUID']} as is_generative: {is_generative} and generative_type: {generative_type}")

        if index % 10 == 0:
            print(f"Processed {index} rows")
            file_df.to_csv("data/classified_data.csv", index=False)

    # save the dataframe to a new csv file
    file_df.to_csv(f"data/classified_data_{model}.csv", index=False)



if __name__=='__main__':
    # print(chat_with_model('llama3.1:8b', 'You are a helpful assistant', 'How are you today?'))
    # read the first 5 rows
    file_df = csvparser.get_csv("data/classified_data.csv")

    classify_paper(
        file_df=file_df,
        system_prompt=SYSTEM_PROMPT_TEMPLATE,
        user_prompt=USER_PROMPT_TEMPLATE,
        model='llama3.1:8b'
    )
