from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = Ollama(model="llama2")

# Running llama2
# print(llm.invoke("how can langsmith help with testing?"))

# Guiding response with a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

print(chain.invoke({"input": "how can langsmith help with testing?"}))
