from pathlib import Path
from typing import Any, List

from pydantic import BaseModel, Field

from langchain.callbacks import get_openai_callback

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import Chroma
from langchain.retrievers import MultiQueryRetriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseOutputParser

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from dotenv import load_dotenv
load_dotenv()


class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
        lines: List[str] = Field(description="Lines of text")

class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


class StripParser(BaseOutputParser):
    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip()

class Config:
    KB_DIR = KB_DIR = Path().absolute().joinpath("KB")
    EMBEDDING = OpenAIEmbeddings(
        deployment="embedding"
    )
    LLM = AzureChatOpenAI(deployment_name="chat", temperature=0)
    OUTPUT_PARSER = LineListOutputParser()

class AgenPintar:
    prompt_system = """
    You are {agen_name} an {vendor} customer support agent, an intelligent chatbot designed to help users answer questions related to {vendor} only. Answer the question with formal and structured Indonesian Language and within the maximum of 500 characters. If the information is not provided, apologize and ask for another service you can provide. 
    
    ```{documents}```
    """

    prompt_system = """
    Assistant name is {agen_name}, Assistant helps the Customers with related question company {vendor}.\n    Be brief in your answers.\n    Reply in Bahasa.\n    ONLY Answer Product {vendor} in Cognitive Search.\n    If there isn't enough information below, please continue with the context of the question.\n    Do not generate answers that don't use the sources below.\n    Do not generate answers from public sources, just only answers from documents below.\n    NEVER ANSWER RELATED QUESTION ABOUT COMPETITORS.\n    If asking a clarifying question to the user would help, ask the question.\n    If the question is not in Bahasa, answer in the language used in the question.\n    If Assistant cannot understand question from Customer, and not have information in Cognitive Search, Give flaging in responses 'NOK' from Backend.\n    If Customers say Hay [GREETING], Assistant can Reply 'Halo, saya {agen_name} siap membantu anda terkait {vendor}'\n    If Customers say [CLOSING, Terimakasih, thanks, etc], Assistant can Reply 'Terimakasih telah menghubungi {agen_name}, sampai jumpa kembali', Please Reply in Variatif Answer Closing\n    Answer as simple as possible\n    Answer ONLY with the facts listed in the list of sources below.  \n

    ```{documents}```
"""

    human_prompt = "{question} please answer in json format and provide intent: <Pembelian, Informasi, Komplain>, sentiment: <Positive, Neutral, Negative>, isData: <default True but if the message or question is not in the data or out of topic or the message isData evaluates to False> and your answer as response_chat in that json"

    def __init__(self, vendor, agen_name) -> None:
        self.vendor = vendor
        self.agen_name = agen_name
        self.persistant_db = Config.KB_DIR / vendor.upper()
    
    def _retrieve_from_docs(self, q: str):
        VECTOR_DB = Chroma(persist_directory=str(self.persistant_db), embedding_function=Config.EMBEDDING)
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate two 
                    different versions of the given user question to retrieve relevant documents from a vector  database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions seperated by newlines. Original question: {question}""",
        )
        llm_chain = LLMChain(llm=Config.LLM, prompt=QUERY_PROMPT, output_parser=Config.OUTPUT_PARSER)

        retriever = MultiQueryRetriever(
            retriever=VECTOR_DB.as_retriever(), llm_chain=llm_chain, parser_key="lines", verbose=True
        )

        unique_docs = retriever.get_relevant_documents(
            query=q
        )
        return unique_docs
    
    def __repr__(self) -> str:
        return f"An Intelligence Agent named {self.agen_name} who work for {self.vendor}"

    def _run(self, q: str):
        docs = self._retrieve_from_docs(q)
        # print(docs)
        parser = StripParser()
        system = SystemMessagePromptTemplate.from_template(self.prompt_system)
        human = HumanMessagePromptTemplate.from_template(self.human_prompt)
        chat_prompt = ChatPromptTemplate.from_messages([system, human])

        llm_chain = LLMChain(llm=Config.LLM, prompt=chat_prompt, output_parser=parser)
        with get_openai_callback() as cb:
            result = llm_chain.run(
                question=q,
                documents=docs, 
                agen_name=self.agen_name, 
                vendor=self.vendor
            )
            print(cb.total_tokens)
            print(cb.completion_tokens)
            print(cb.total_cost)
        return result
    
    def __call__(self, q: str) -> str:
        return self._run(q)
    

if __name__ == "__main__":
    import time
    agen = AgenPintar("PosIndo", "vida")
    print("ready gan")
    while True:
        st = time.time()
        q = input()
        r = agen(q)
        print("runtime: ", time.time() - st)
