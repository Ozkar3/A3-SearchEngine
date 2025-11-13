
'''
    1. Get documents to be indexed & extract text
    2. Get Tokens
    3. Text Process Tokens (normalization,stemming,stopwords)
    4. Build Indexer
    5. Save Indexer to Disk
    6. Print report
 '''

import re
import os                               # crawling file directories 
import json                             # loading the json files
from bs4 import BeautifulSoup           # parsing HTML
from collections import Counter         # for frequency of words
from nltk.stem import PorterStemmer     # porter Stemmer

stemmer = PorterStemmer()
IMPORTANCE_WEIGHT = 10                  # weight of important words (10 is arbituary)
FILE_LIMIT = 1000

# DATA STRUCTURES
inverted_index = {}
doc_id_map = {}
doc_id = 0
# ----------------
#indexed_documents = 0

for root, dirs, files in os.walk("DEV"):
    # might delete this later
    # just makes the crawling not random and goes through each file in the actual order of the files
    # instead of randomly opening files in any order; easier visualise and debug 
    dirs.sort()
    files.sort()
    # ---------------------------------------------

    for file in files:
        if file.endswith(".json"):
            if doc_id >= FILE_LIMIT:
                break
            
            file_path = os.path.join(root, file)
            print(f"***** Crawling {file_path} (DocID: {doc_id}) *****")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                url = data.get("url", "No URL found")
                html_content = data.get("content", "")
                
                if not html_content:
                    print(f"***** Content not found ***** {url}")
                    continue
                
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Getting IMPORTANT text
                important_text = ""
                for tag in soup.find_all(['title', 'h1', 'h2', 'h3', 'b', 'strong']):
                    important_text += " " + tag.get_text()
                
                # Getting REGULAR text
                regular_text = soup.get_text()
                
                # Tokenize (alphanumeric words)
                regular_tokens = re.findall(r'\w+', regular_text.lower())
                important_tokens = re.findall(r'\w+', important_text.lower())
                
                # Stemming
                stemmed_regular = [stemmer.stem(token) for token in regular_tokens]
                stemmed_important = [stemmer.stem(token) for token in important_tokens]

                # Word Frequency
                regular_tf = Counter(stemmed_regular)
                important_tf = Counter(stemmed_important)
                
                final_tf = regular_tf 
                """
                for regular words it stores the frequency of the words to the index
                for important words it stores in the index: (frequency in the body + (frequency in important tags * 10))
                """
                for token, freq in important_tf.items():
                    # If only in important words
                    if token not in final_tf:
                        final_tf[token] = freq * IMPORTANCE_WEIGHT
                    # If also in regular text add more to score
                    else:
                        final_tf[token] += freq * IMPORTANCE_WEIGHT
                                
                # Keeps track of doc id to url for later use
                doc_id_map[doc_id] = url
                
                for token, tf in final_tf.items():
                    posting = {"doc_id": doc_id, "tf": tf}
                    if token not in inverted_index:
                        inverted_index[token] = [posting]
                    else:
                        inverted_index[token].append(posting)
                doc_id += 1 
            
            except Exception as e:
                print(f"  ERROR getting file {file_path}: {e}")
    
    if doc_id >= FILE_LIMIT:
        break

print(f"\n***** Processing Finished *****")

print("\nSaving to disk...")
with open("inverted_index.json", "w") as f:
    json.dump(inverted_index, f)

with open("doc_id_map.json", "w") as f:
    json.dump(doc_id_map, f)

print("\nDone saving index and doc_id_map to disk...")


print("\n*************** M1 REPORT ***************")
print(f"1. Number of indexed documents: {doc_id}")
print(f"2. Number of unique tokens: {len(inverted_index)}")
index_size_kb = os.path.getsize("inverted_index.json") / 1024       # /1024 is to convert bytes to kb
print(f"3. Total size of index on disk: {index_size_kb:.2f} KB")
