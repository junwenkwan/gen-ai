from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import initialize_agent, Tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

prompt = hub.pull("hwchase17/openai-tools-agent")

search = GoogleSerperAPIWrapper()

search_tool = Tool(
                    name="Google Serper",
                    func=search.run,
                    description="useful for when you need to ask with search"
                )

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

wikipedia_tool = Tool(
                    name="Wikipedia",
                    func=wikipedia.run,
                    description="wikipedia"
                )

tools = [search_tool, wikipedia_tool]

agent = initialize_agent(tools, llm, verbose=True)

results = agent.invoke(
    {
        "input": "What is the relationship between Applied EV and Suzuki? Please elaborate.",
    }
)

print(results['output'])