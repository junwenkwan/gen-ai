from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

prompt = hub.pull("hwchase17/openai-tools-agent")

search = DuckDuckGoSearchResults()

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [search, wikipedia]

agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke(
    {
        "input": "What is the relationship between Applied EV and Suzuki?",
    }
)