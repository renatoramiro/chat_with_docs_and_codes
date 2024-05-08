from langchain_community.document_loaders import (WebBaseLoader, DirectoryLoader)
from langchain_text_splitters import (Language, RecursiveCharacterTextSplitter)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

import os
from dotenv import load_dotenv

load_dotenv() # take environment variables from .env.

CODE_PATH = os.getenv('CODE_PATH')
CHROMADB_PATH = os.getenv('CHROMADB_PATH')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')
OLLAMA_EMBEDDING = os.getenv('OLLAMA_EMBEDDING')


urls = ['https://docs.crewai.com/core-concepts/Agents/', 'https://docs.crewai.com/core-concepts/Tasks/',
        'https://docs.crewai.com/core-concepts/Tools/', 'https://docs.crewai.com/core-concepts/Processes/',
        'https://docs.crewai.com/core-concepts/Crews/', 'https://docs.crewai.com/core-concepts/Collaboration/',
        'https://docs.crewai.com/core-concepts/Memory/', 'https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/',
        'https://docs.crewai.com/how-to/Create-Custom-Tools/', 'https://docs.crewai.com/how-to/Sequential/',
        'https://docs.crewai.com/how-to/Hierarchical/', 'https://docs.crewai.com/how-to/LLM-Connections/',
        'https://docs.crewai.com/how-to/Customizing-Agents/', 'https://docs.crewai.com/how-to/Human-Input-on-Execution/']

header_template={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',}

# LOAD CREAWAI DOCS
web_loader = WebBaseLoader(urls, header_template=header_template)
docs = web_loader.load()

# LOAD CREWAI CODE EXAMPLES
code_loader = DirectoryLoader(CODE_PATH, glob='**/*.py')
codes = code_loader.load()

# SPLIT TEXT
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
text_chunks = text_splitter.split_documents(docs)

# SPLIT CODE
code_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=200, chunk_overlap=30)
code_chunks = code_splitter.split_documents(codes)

embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING, show_progress=False)

# VECTOR DATABASE
persist_directory = CHROMADB_PATH 
vectordb = Chroma.from_documents(documents=text_chunks,
                      embedding=embeddings,
                      persist_directory=persist_directory)

vectordb.add_documents(code_chunks)
vectordb.persist()

from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)
retriever = vectordb.as_retriever(search_kwargs={'k': 10})

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
new_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_system_prompt = """You are a senior python developer for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(new_retriever, question_answer_chain)

from langchain_core.messages import HumanMessage

question = ''
chat_history = []

while question != 'qq':
    question = input('Ask something: ')

    if question == 'qq':
        print('Bye Bye')
    else:
        result = rag_chain.invoke({"input": question, "chat_history": chat_history}) 
        chat_history.extend([HumanMessage(content=question), result['answer']])
        print('\n** ANSWER **')
        print(result['answer'])
        print('********\n')
