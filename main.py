import os
import time
from tqdm import tqdm
from collections import defaultdict

import pandas as pd
import requests
from dotenv import load_dotenv
from variables import *
import csvparser
from colorama import init, Fore, Back, Style

init(autoreset=True)  # Initialize colorama

# ASCII Art Banner
BANNER = r"""
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   ███╗   ██╗███████╗██████╗ ██╗   ██╗██╗      █████╗  ║
║   ████╗  ██║██╔════╝██╔══██╗██║   ██║██║     ██╔══██╗ ║
║   ██╔██╗ ██║█████╗  ██████╔╝██║   ██║██║     ███████║ ║
║   ██║╚██╗██║██╔══╝  ██╔══██╗██║   ██║██║     ██╔══██║ ║
║   ██║ ╚████║███████╗██████╔╝╚██████╔╝███████╗██║  ██║ ║
║   ╚═╝  ╚═══╝╚══════╝╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝ ║
║                                                       ║
║            by NETWORK INSTITUTE                       ║
╚═══════════════════════════════════════════════════════╝
"""


load_dotenv('.env')

NEBULA_URL = 'https://nebula.cs.vu.nl/api/chat/completions'
NEBULA_TOKEN = str(os.getenv('NEBULA_TOKEN'))


def spinner(seconds):
    """Display a spinner animation while waiting"""
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    for _ in range(int(seconds * 10)):
        for char in chars:
            print(f"\r{Fore.CYAN}{char} Processing...{Style.RESET_ALL}", end='', flush=True)
            time.sleep(0.1)
    print()


def chat_with_model(model, system_prompt, user_prompt):
    print(f"\n{Fore.BLUE}⚡ Sending request to {Fore.YELLOW}{model}{Fore.BLUE}...{Style.RESET_ALL}")
    spinner(1)  # Visual spinner while waiting

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
    print(f"{Fore.GREEN}✅ Request completed successfully!{Style.RESET_ALL}")
    return response.json()['choices'][0]['message']['content']


def classify_paper(file_df, system_prompt, user_prompt, model):
    # Display ASCII art banner
    print(f"{Fore.CYAN}{BANNER}{Style.RESET_ALL}")
    print(
        f"\n{Fore.MAGENTA}★彡 Starting paper classification with {Fore.YELLOW}{model} {Fore.MAGENTA}彡★{Style.RESET_ALL}\n")

    # Create stats counters
    total = len(file_df)
    classified = 0
    skipped = 0
    errors = 0

    # For batch reporting of skipped papers
    skipped_batch = []
    skipped_count = 0
    BATCH_SIZE = 100  # Report every 100 skipped papers

    # To track different types of skips
    skip_reasons = defaultdict(int)

    # Create progress bar
    progress_bar = tqdm(total=total, desc=f"{Fore.GREEN}📑 Classifying Papers",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    # iterate through file_df and give prompt + abstract
    for index, row in file_df.iterrows():
        # Update the progress bar postfix to show current stats
        progress_bar.set_postfix_str(f"Classified: {classified} | Skipped: {skipped} | Errors: {errors}")

        # if the classification has already been made, skip
        if row["generative_type"] in GENERATIVE_TYPES:
            skipped += 1
            skip_reasons["already_classified"] += 1
            progress_bar.update(1)
            continue

        abstract = row['Abstract']
        title_and_subtitle = str(row['Title']) + " " + str(row["Subtitle"])
        current_user_prompt = user_prompt.format(title=title_and_subtitle, abstract=abstract)

        if pd.isna(abstract) or abstract == '' or abstract == 'nan' or abstract == 'None' or len(
                abstract) < 5 or abstract == 'N/A':
            # Instead of printing each skipped paper, collect them
            skipped += 1
            skipped_count += 1
            skip_reasons["invalid_abstract"] += 1
            skipped_batch.append(row['UUID'])

            # Only print a batch warning when we reach the batch size
            if skipped_count >= BATCH_SIZE:
                print(f"{Fore.YELLOW}⚠ Skipped {skipped_count} papers with invalid abstracts{Style.RESET_ALL}")
                skipped_count = 0
                skipped_batch = []

            progress_bar.update(1)
            continue

        llm_classification = chat_with_model(
            model=model,
            system_prompt=system_prompt,
            user_prompt=current_user_prompt
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

            # Add visual indicators based on classification
            gen_indicator = f"{Fore.GREEN}✓ GENERATIVE" if is_generative else f"{Fore.RED}✗ NON-GENERATIVE"
            print(
                f"{Fore.CYAN}🔍 Paper {row['UUID']}: {gen_indicator} {Fore.YELLOW}[Type: {generative_type}]{Style.RESET_ALL}")
            classified += 1
        except Exception as e:
            # Keep detailed error messages as they're important for debugging
            print(f"{Fore.RED}❌ Error for {row['UUID']}: {str(e)}{Style.RESET_ALL}")
            errors += 1
            progress_bar.update(1)
            continue

        progress_bar.update(1)

        if index % 10 == 0 and index > 0:
            # Make save messages less frequent but still informative
            print(f"{Fore.CYAN}💾 Saving progress ({index}/{total} papers){Style.RESET_ALL}")
            file_df.to_csv("data/classified_data.csv", index=False)

    progress_bar.close()

    # Print any remaining skipped papers in the batch
    if skipped_count > 0:
        print(f"{Fore.YELLOW}⚠ Skipped {skipped_count} papers with invalid abstracts{Style.RESET_ALL}")

    # save the dataframe to a new csv file
    file_df.to_csv(f"data/classified_data_{model}.csv", index=False)

    # Add skip reason breakdown to the summary
    skip_reason_text = ""
    for reason, count in skip_reasons.items():
        skip_reason_text += f"\n║ {Fore.YELLOW}  • {reason}: {Fore.WHITE}{count:<19}{Style.RESET_ALL} ║"

    # Final stats with fancy box
    print(f"""
╔═════════════════════════════════════════╗
║ {Fore.GREEN}📊 Classification Results Summary 📊{Style.RESET_ALL}     ║
╠═════════════════════════════════════════╣
║ {Fore.CYAN}✦ Total papers processed: {Fore.WHITE}{total:<13}{Style.RESET_ALL} ║
║ {Fore.GREEN}✦ Successfully classified: {Fore.WHITE}{classified:<13}{Style.RESET_ALL} ║
║ {Fore.YELLOW}✦ Skipped papers: {Fore.WHITE}{skipped:<19}{Style.RESET_ALL} ║{skip_reason_text}
║ {Fore.RED}✦ Errors encountered: {Fore.WHITE}{errors:<16}{Style.RESET_ALL} ║
╚═════════════════════════════════════════╝
    """)

    print(f"{Fore.MAGENTA}🎉 Classification complete! Results saved to:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📁 data/classified_data_{model}.csv{Style.RESET_ALL}")


if __name__ == '__main__':
    print(f"\n{Fore.YELLOW}⭐ Initializing Paper Classification System ⭐{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📂 Loading data...{Style.RESET_ALL}")

    # Loading animation
    spinner(2)

    file_df = csvparser.get_csv("data/classified_data.csv")

    print(f"{Fore.GREEN}✅ Data loaded successfully! Found {len(file_df)} records.{Style.RESET_ALL}")

    classify_paper(
        file_df=file_df,
        system_prompt=SYSTEM_PROMPT_TEMPLATE,
        user_prompt=USER_PROMPT_TEMPLATE,
        model='llama3.1:8b'
    )

    # Final success message with ASCII art
    print(f"""
{Fore.GREEN}
   ____                      _      _       _ 
  / ___|___  _ __ ___  _ __ | | ___| |_ ___| |
 | |   / _ \| '_ ` _ \| '_ \| |/ _ \ __/ _ \ |
 | |__| (_) | | | | | | |_) | |  __/ ||  __/_|
  \____\___/|_| |_| |_| .__/|_|\___|\__\___(_)
                      |_|                     
{Style.RESET_ALL}
    """)
