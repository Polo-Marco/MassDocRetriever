import json
import os
import pickle
from multiprocessing import Pool, cpu_count

from langchain.docstore.document import Document
from tqdm import tqdm

from utils.text_utils import text_preprocess


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
            doc_level.append(
                Document(
                    page_content=text_preprocess(text),
                    metadata={"id": text_preprocess(doc_id)},
                )
            )
            # Process sentence-level docs (skip empty lines)
            for sline in entry["lines"].split("\n"):
                if not sline.strip():
                    continue  # Skip fully empty lines
                try:
                    idx, sentence = sline.split("\t", 1)
                    if sentence.strip():  # Only add if sentence is not empty
                        sentence_level.append(
                            Document(
                                page_content=text_preprocess(sentence.strip()),
                                metadata={
                                    "doc_id": text_preprocess(doc_id),
                                    "line_id": idx,
                                },
                            )
                        )
                except ValueError:
                    continue  # Skip malformed lines (e.g., no tab character)
    return doc_level, sentence_level


def process_one_file(file):
    doc_temp, sent_temp = prepare_doc_and_sentence_indexes(
        os.path.join(file_path, file)
    )
    return doc_temp, sent_temp


if __name__ == "__main__":
    cpu_count = 1  # for multi process
    doc_level_docs, sentence_level_docs = [], []
    file_path = "./data//wiki-pages/"
    files = [f for f in os.listdir(file_path) if f.endswith(".jsonl")][:1]
    with Pool(processes=cpu_count) as pool:
        results = list(tqdm(pool.imap(process_one_file, files), total=len(files)))
    for doc_temp, sent_temp in results:
        doc_level_docs.extend(doc_temp)
        sentence_level_docs.extend(sent_temp)
    print(
        f"Total: {len(doc_level_docs)} documents, {len(sentence_level_docs)} sentences loaded."
    )
    print("Saving into files...")
    with open("./data/doc_level_docs_dev.pkl", "wb") as f:
        pickle.dump(doc_level_docs, f)
    with open("./data/sentence_level_docs_dev.pkl", "wb") as f:
        pickle.dump(sentence_level_docs, f)
    print("Done!")
