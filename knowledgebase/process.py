
# For upload document

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from pypdf import PdfReader

from langchain.document_loaders import BSHTMLLoader
# from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings


EMBEDDING = OpenAIEmbeddings(
    deployment="embedding"
)

KB_DIR = Path().absolute().joinpath("KB")
SPLITTER = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)


def chunking(pages):
    splits = SPLITTER.split_documents(pages)
    # print(len(splits))
    return splits


# save to db
def save_to_db(vendor, splits, n=8):
    persist_dir = KB_DIR.joinpath(vendor.upper())
    persist_dir.mkdir(exist_ok=True, parents=True)

    for i in range(0, len(splits), n):
        Chroma.from_documents(
            documents=splits[i : i+n], 
            embedding=EMBEDDING, 
            persist_directory=str(persist_dir)
        )
        print(f"processing {i}/{len(splits)}", end="\r")
    return True
    

def load_pdf(file):
    loader = PdfReader(file)
    pages = list()
    metadata = list()
    # Read the text from each page
    for idx, page in enumerate(loader.pages):
        text = page.extract_text()
        metadata.append({"documents": idx})
        pages.append(text)
    # pages = loader.load_and_split()
    # print(pdf_text)
    metadatas = SPLITTER.create_documents(texts=pages)
    return metadatas


def load_htlm(url):
    loader = BSHTMLLoader(url)
    data = loader.load()
    return data
    

def proc(files, vendor):
    pages = load_pdf(files)
    splits = chunking(pages)
    return save_to_db(vendor, splits)
    




if __name__ == "__main__":
    from argparse import ArgumentParser
    args = ArgumentParser()
    args.add_argument("--files")
    args.add_argument("--vendor")

    args = args.parse_args()
    proc(args.files, args.vendor)


