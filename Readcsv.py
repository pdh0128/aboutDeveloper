from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def csv_agent_executor(input):
    csv_agent = create_csv_agent(llm=llm, path="/Users/pdh/Desktop/프로젝트/코드인터프리터/survey_results_public.csv", verbose=True, allow_dangerous_code=True, prefix="출력은 항상 한글로 해줘")

    FORMAT_INSTRUCTIONS = """Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do, what action to take
    Action: python_repl_ast
    Action Input: the input to the action, never add backticks "`" around the action input
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    """

    query = input + FORMAT_INSTRUCTIONS
    res = csv_agent.invoke(query)

    return res['output']

if __name__ == "__main__":
    csv_agent_executor("개발자들의 평균 나이는 몇이야?")