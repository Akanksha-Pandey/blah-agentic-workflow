from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from tool_get_current_time import get_current_time

# 1. Load local LLM
llm = Ollama(model="qwen3:8b")

# 2. Register tools
tools = [
    Tool(
        name="GetCurrentTime",
        func=get_current_time,
        description="Returns the current system time"
    )
]

# 3. Create agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 4. Run agent
result = agent.run(
    "Say hello politely and tell me the current time."
)

print("\nFINAL ANSWER:")
print(result)
