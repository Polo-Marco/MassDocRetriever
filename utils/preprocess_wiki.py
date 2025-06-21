from langchain.docstore.document import Document
import json
from tqdm import tqdm
import os
import pickle
from utils import clean_spacing, strF2H, strF2H_w_punctuation
from multiprocessing import Pool, cpu_count
def text_preprocess(text):
    '''preprocess for half characters'''
    return clean_spacing(strF2H_w_punctuation(text))
def prepare_doc_and_sentence_indexes(jsonl_path):
    doc_level = []
    sentence_level = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            doc_id = entry["id"]
            text = entry["text"]
            doc_level.append(Document(page_content=text_preprocess(text), 
                                      metadata={"id": text_preprocess(doc_id)}))
            # Process sentence-level docs (skip empty lines)
            for sline in entry["lines"].split("\n"):
                if not sline.strip():
                    continue  # Skip fully empty lines
                try:
                    idx, sentence = sline.split("\t", 1)
                    if sentence.strip():  # Only add if sentence is not empty
                        sentence_level.append(Document(
                            page_content=text_preprocess(sentence.strip()),
                            metadata={"doc_id": text_preprocess(doc_id), "line_id": idx}
                        ))
                except ValueError:
                    continue  # Skip malformed lines (e.g., no tab character)
    return doc_level, sentence_level

# Example usage:
# doc_temp, sent_temp= prepare_doc_and_sentence_indexes("./wiki-pages/wiki-001.jsonl")
# print(doc_temp[:5])
# #print(sent_temp[5:10])
# print(f"{len(doc_temp)} documents loaded, {len(sent_temp)} sentences loaded.")
# exit()

def process_one_file(file):
    doc_temp, sent_temp = prepare_doc_and_sentence_indexes(os.path.join(file_path, file))
    return doc_temp, sent_temp

if __name__ == "__main__":
    cpu_count = 10
    doc_level_docs, sentence_level_docs = [], []
    file_path = "./wiki-pages/"
    files = [f for f in os.listdir(file_path) if f.endswith('.jsonl')]
    with Pool(processes=cpu_count) as pool:
        results = list(tqdm(pool.imap(process_one_file, files), total=len(files)))
    for doc_temp, sent_temp in results:
        doc_level_docs.extend(doc_temp)
        sentence_level_docs.extend(sent_temp)
    print(f"Total: {len(doc_level_docs)} documents, {len(sentence_level_docs)} sentences loaded.")
    print("Saving into files...")
    with open("doc_level_docs.pkl", "wb") as f:
        pickle.dump(doc_level_docs, f)
    with open("sentence_level_docs.pkl", "wb") as f:
        pickle.dump(sentence_level_docs, f)
    print("Done!")

# # Usage example
# doc_level_docs, sentence_level_docs = prepare_doc_and_sentence_indexes("your_data.jsonl")
# print(f"{len(doc_level_docs)} docs loaded, {len(sentence_level_docs)} sentences loaded.")
# id2doc = {doc.metadata['id']: doc for doc in doc_level_docs}

# # Query by id:
# doc_id = "數學"
# doc = id2doc.get(doc_id)
# if doc:
#     print(doc.page_content)   # The full text of the document
#     print(doc.metadata)       # The metadata dictionary
# else:
#     print("Document not found.")

#     sentence_key2doc = {
#     (doc.metadata['doc_id'], doc.metadata['line_id']): doc
#     for doc in sentence_level_docs
# }
# print(sentence_level_docs[:5])
# Query by doc_id and line_id
# key = ("your_doc_id_here", "your_line_id_here")
# sentence_doc = sentence_key2doc.get(key)
# if sentence_doc:
#     print(sentence_doc.page_content)    # The sentence text
#     print(sentence_doc.metadata)        # Metadata: doc_id, line_id, doc_text, etc.
# else:
#     print("Sentence not found.")