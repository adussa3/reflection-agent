###########
# IMPORTS #
###########

# TypedDict is a type dictionary which creates a structured dictionary with hints for keys and values,
# which can be used for type checking and better code readability
#
# We'll use this to define our state schema for our LangGraph graph
# We need this because LangGraph needs types state definitions to know which data flows in to and out of the graph

# Annotated is a way to add metadata to types hints, which can be used for validation, serialization, etc.
from typing import TypedDict, Annotated

# load_dotenv takes all the environment variables from the .env file and adds them to the environment variables
# of the system. This allows us to access these variables in our code using os.environ or other similar methods.
from dotenv import load_dotenv
load_dotenv()

# The BaseMessage is the abstract base class for all message types in LangChain
# We're going to be using it as a type hint for the messages list in our state schema
# This ensures saftey for different message types (HumanMessage, AIMessage, SystemMessage, etc.)
#
# The HumanMessage represents a message sent by a user
# We want this because for specific messages, we want to distingish between the user content from the ai responses
from langchain_core.messages import BaseMessage, HumanMessage

# END is a constant that hold __end__, which is the key for LangGraph's default ending node
# When we reach the END node with this key, the LangGraph stops execution
#
# the StateGraph is the main class for building stateful LangGraph graphs
# When we create our workflow (which descriibes the execution of nodes and edges of our agentic flow)
# we need to provide the flow with our state
#
# The state is simply going ot be a data structure (usually a dictionary or pydantic class) which is maintained
# for the entire execution, and it holds the information of the execution
#
# We can store their intermediate results, LLM responses, and basically everything we can think of we can store
# Every node we run will have access to this state (it's the input for every node)
# and the nodes can also modify and update the state (by returning a new state or modifying the existing state)
from langgraph.graph import END, StateGraph

# add_messages is a LangGraph reducer function that APPENDS to the existing messages list in the state, instead of replacing it
# This is very useful for our use case because we want to maintain a history of all the messages that we had with the LLM
from langgraph.graph.message import add_messages

# Import the chains that we created in the chains module
from chains import generation_chain, reflection_chain

#######################
# DEFINE STATE SCHEMA #
#######################

# This is the data structure that every node in our graph will have access to
# Note: this is NOT the imported MessageGraph from langgraph.graph, this is just a custom state schema that we created for our graph
# NOTE: the state is simply a dictionary that holds all the information about the execution, and it can be modified and updated by every node in the graph
class MessageGraph(TypedDict):
  # The messages list will hold all the messages that were generated so far by us and the LLM
  # and this will be the "history" that we feed into the MessagesPlaceholder in our prompts
  #
  # We want to append messages to the messages list in our state, instead of replacing them
  # (1) Annotated is metadata that tells LangGraph how to handle state updates. In this case, it's telling LangGraph to use the add_messages reducer function
  # (2) add_messages is a reducer function that tells LangGraph to append to the existing messages list, instead of replacing it
  #
  # Note: a reducer function is general terminology in LangGraph which tells it how to update the state
  # This is LangGraph's key advantages when it comes to flexibility
  messages: Annotated[list[BaseMessage], add_messages]

#########
# NODES #
#########

# These constants are the keys for our LangGraph nodes that we're going to create
GENERATE = "generate"
REFLECT = "reflect"

# The generation node function will receive a state, of type MessageGraph, as the input
def generation_node(state: MessageGraph):
  # The node runs and invokes the generation chain with the state messages as the input
  # The state is going to hold all of the critiques and previous generations that we had so far
  # 
  # Remember the MessagesPlaceholder, now we're plugging in messages which is simply the state which is the agent's message history
  #
  # Once we get the response back from the LLM, we'll take the return value and append it to the state (which happens under the hood)
  res = generation_chain.invoke({"messages": state["messages"]})
  return {"messages": [res]}

# The reflection node function will receive a state, of type MessageGraph, as the input
def reflection_node(state: MessageGraph):
  # The node runs and invokes the reflection chain with the state messages as the input
  # The state is going to hold all of the critiques and previous generations that we had so far
  #
  # Remember the MessagesPlaceholder, now we're plugging in messages which is simply the state which is the agent's message history
  res = reflection_chain.invoke({"messages": state["messages"]})

  # Once we get the response back from the LLM, we change it into a HumanMessage (instead of the default AIMessage)
  # We do this because we want to trick the LLM that the human is sending the critique
  # so it seems like we're having a conversation between the human and the assistant
  # This is a very important technique when we implement things with LangGraph
  #
  # NOTE: we need to return a dictionary with the key "messages" so StateGraph knows how to update the state
  # and then we return the value and append it to the state (which happens under the hood)
  return {"messages": [HumanMessage(content=res.content)]}

#########
# GRAPH #
#########

# Initialize Graph with our custom MessageGraph state schema
# This tells LanGraph what's our state and how to update it
builder = StateGraph(state_schema=MessageGraph)

# Add Generate Node
builder.add_node(GENERATE, generation_node)

# Add Reflection Node
builder.add_node(REFLECT, reflection_node)

# Set the starting node to be the generation node
builder.set_entry_point(GENERATE)

# This is a conditional edge function that takes the state and returns the key of the next node to execute
def should_continue(state: MessageGraph):
  if len(state["messages"]) > 6:
    return END
  return REFLECT

# Add conditional edge to the generation node
# add_conditional_edges is a routing function that tells LangGraph which node to execute next based on the return value of the should_continue function
# The 3rd argument is a path-mapping dictionary which maps the strings to the actual nodes in the graph
# NOTE: the 3rd argument is optional, but when you visualize the graph, it doesn't the conditional edge paths, so it's good to add it for better visualization
#
# NOTE: conditional edges do NOT return the update state like nodes, it only returns the key of the next node to execute
builder.add_conditional_edges(GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT})

# Add edge from the reflection node back to the generation node
builder.add_edge(REFLECT, GENERATE)

# Compile graph
graph = builder.compile()

###################
# VISUALIZE GRAPH #
###################

# This prints the graph visualization in the console using mermaid syntax
# We can paste this mermaid code in the mermaid live editor (https://mermaid.live/) to see the graph visualization
print(graph.get_graph().draw_mermaid())

# This prints the graph visualization in the console using ascii characters
# Note: you need to install Gandalf to view the graph visualization
print(graph.get_graph().draw_ascii())

# This saves the graph visualization as a png file
with open("new reflection graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

################
# INVOKE GRAPH #
################

if __name__ == '__main__':
  print("Hello LangGraph")

  inputs = HumanMessage(content="""Make this tweet better:"
                                   @LangChainAI
          - newly Tool Calling feature is seriously underrated.
                        
          After a long wait, it's here- making the implementation of agents across different with function calling - super easy.
                        
          Made a video covering their newest blog post
  """)

  response = graph.invoke(inputs)