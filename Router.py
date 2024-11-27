from typing import Any

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from coded import agent_executor
from Readcsv import csv_agent_executor
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub



load_dotenv()

def router_excutoer(input):
    def python_agent_executor_wrapper(original: str) -> dict[str, Any]:
        return agent_executor.invoke(input={"input": original})

    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="파이썬 코드를 작성할 때 유용한 도구입니다. 입력으로 코드를 제공하지 않아도 작동합니다."
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_executor,
            description="개발자와 관련된 정보(공부 방법, 기술, 이력 등)에 대한 질문에 사용합니다."
        )
    ]
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    promps = base_prompt.partial(instructions="CSV Agent는 개발자 정보에 대한 질문에 사용되어야 하며, 그 외에는 Python Agent를 사용하십시오. 개발자 혹은 개발과 연관이 없는 질문에는 '모른다'고 응답하세요.")
    control_agent = create_react_agent(
        prompt=promps,
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        tools=tools,
    )
    control_agent_executor = AgentExecutor(agent=control_agent, tools=tools, verbose=True)
    res = control_agent_executor.invoke(input={"input": input})
    return res['output']