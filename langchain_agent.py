import os
os.environ["OPENAI_API_KEY"] = "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"

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

prompt = hub.pull("hwchase17/openai-tools-agent")

search = DuckDuckGoSearchResults()

tools = [search]

agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke(
    {
        "input": "What is the relationship between Applied EV and Suzuki?",
        "chat_history": [
            HumanMessage(content="hi! my name is bob"),
            AIMessage(content="Hello Bob! How can I assist you today?"),
        ],
    }
)