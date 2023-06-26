import json
import os
from goose3 import Goose
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

g = Goose({'use_meta_language': False, 'target_language':'pt'})

url = 'URL'

site = g.extract(url)
content = site.cleaned_text

with open('env.json') as env:
  data = json.load(env)

api_key = data['api_key']
os.environ["OPENAI_API_KEY"] = api_key

text_splitter = CharacterTextSplitter(
  separator = "\n",
  chunk_size = 1000,
  chunk_overlap  = 200,
  length_function = len,
)

texts = text_splitter.split_text(content)

embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")

while True:
  query = input("Ask a question... ")
  if query:
    docs = docsearch.similarity_search(query)
    answer = chain.run(input_documents=docs, question=query)
    print(answer)
  else:
    break