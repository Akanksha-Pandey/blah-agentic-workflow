from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
import operator
from tool_get_current_time import get_current_time

# 1. Load local LLM
llm = OllamaLLM(model="qwen3:8b")

# 2. Define the state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# 3. Register tools
@tool
def get_current_time_tool():
    """Returns the current system time"""
    return get_current_time()

tools = [get_current_time_tool]
tools_dict = {tool.name: tool for tool in tools}

# 4. Define nodes
def model_node(state: AgentState):
    """LLM node that generates responses"""
    messages = state["messages"]
    
    # Create a prompt with tools information
    tools_info = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
    system_prompt = f"""You are a helpful assistant. You have access to the following tools:
{tools_info}

When you need to use a tool, respond with the tool name in square brackets like [tool_name]."""
    
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response)]}

def tool_node(state: AgentState):
    """Tool execution node"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if we should call a tool based on message content
    if "time" in last_message.content.lower():
        result = get_current_time_tool.invoke({})
        return {"messages": [ToolMessage(content=str(result), tool_call_id="get_current_time_tool")]}
    
    return {"messages": []}

# 5. Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("model", model_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("model")
workflow.add_edge("model", "tools")
workflow.add_edge("tools", END)

agent = workflow.compile()

# 6. Run the agent
initial_state = {"messages": [HumanMessage(content="What is the current time?")]}
result = agent.invoke(initial_state)
print("\n=== Agent Result ===")
for msg in result["messages"]:
    print(f"{msg.__class__.__name__}: {msg.content}")
