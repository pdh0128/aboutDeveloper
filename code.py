from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool

from dotenv import load_dotenv

load_dotenv()

instruction = """
너는 질문에 대해 Python 코드를 작성하기 실행하기 위해 설계되었어. 
너는 Python REPL에 접근할 수 있고 이를 통해 파이썬 코드를 실행할 수 있어. 
오류가 발생하면 오류를 디버깅하고 다시 시도해. 
질문에 대한 답은 실행 결과를 출력하는 Python 코드로만 제공해.
코드를 실행하지 않고도 답을 알 수 있지만 답을 얻기 위헤서는 코드를 실행해야 해.
Python으로 해결할 수 없는 질문이라면 '모르겠습니다'라고 답해
 """
base_prompt = hub.pull("langchain-ai/react-agent-template")
promps = base_prompt.partial(instructions=instruction)

tools = [PythonREPLTool()]

agent = create_react_agent(prompt=promps, tools=tools, llm=ChatOpenAI(model="gpt-4o-mini", temperature=0))
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
if __name__ == "__main__":
    res = agent_executor.invoke(input={"input" : "bfs 코드 짜줘"})
    print(res)