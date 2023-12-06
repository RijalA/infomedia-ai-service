from pathlib import Path
from typing import List

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import Chroma
from langchain.retrievers import MultiQueryRetriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import Tool

from pydantic import BaseModel, Field


class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
        lines: List[str] = Field(description="Lines of text")

class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


class Config:
    KB_DIR = KB_DIR = Path().absolute().joinpath("KB")
    EMBEDDING = OpenAIEmbeddings(
        deployment="embedding"
    )
    LLM = AzureChatOpenAI(deployment_name="chat", temperature=0)
    OUTPUT_PARSER = LineListOutputParser()

def retrieve_from_docs(q: str):
        VECTOR_DB = Chroma(persist_directory=str(Config.KB_DIR), embedding_function=Config.EMBEDDING)
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five 
                    different versions of the given user question to retrieve relevant documents from a vector  database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions seperated by newlines. Original question: {question}""",
        )
        llm_chain = LLMChain(llm=Config.LLM, prompt=QUERY_PROMPT, output_parser=Config.OUTPUT_PARSER)

        retriever = MultiQueryRetriever(
            retriever=VECTOR_DB.as_retriever(), llm_chain=llm_chain, parser_key="lines"
        )

        unique_docs = retriever.get_relevant_documents(
            query=q
        )
        return "\n".join(unique_docs)