import os
os.environ["OPENAI_API_KEY"] = "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"

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

from langchain import hub

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-tools-agent")

from langchain.agents import create_openai_tools_agent

from langchain_community.tools import DuckDuckGoSearchResults

search = DuckDuckGoSearchResults()

tools = [search]

agent = create_openai_tools_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

text = "What is the relationship between Applied EV and Suzuki?"

agent_executor.invoke({"input": text})
