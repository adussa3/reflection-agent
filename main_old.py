###########
# IMPORTS #
###########

from typing import List, Sequence

# load_dotenv takes all the environment variables from the .env file and adds them to the environment variables
# of the system. This allows us to access these variables in our code using os.environ or other similar methods.
from dotenv import load_dotenv
load_dotenv()

#
from langchain_core.messages import BaseMessage, HumanMessage

# END is a constant that hold __end__, which is the key for LangGraph's default ending node
# When we reach the END node with this key, the LangGraph stops execution
#
# The MessageGraph (deprecated) is a type of graph whose state is simply a sequence of messages
# In this graph, every node will receive a list of messages as input,
# and the output for every node returns one or more messages as output
from langgraph.graph import END, MessageGraph

# Import the chains that we created in the chains module
from chains import generation_chain, reflection_chain

#########
# NODES #
#########

# These constants are the keys for our LangGraph nodes that we're going to create
GENERATE = "generate"
REFLECT = "reflect"

# The generation node function will receive a state as the input
# In a LangGraph's MessageGraph, the state is simply a sequence of messages
def generation_node(state: Sequence[BaseMessage]):
  # The node runs and invokes the generation chain with the state messages as the input
  # The state is going to hold all of the critiques and previous generations that we had so far
  # 
  # Remember the MessagesPlaceholder, now we're plugging in messages which is simply the state which is the agent's message history
  #
  # Once we get the response back from the LLM, we'll take the return value and append it to the state (which happens under the hood)
  return generation_chain.invoke({"messages": state})

# The reflection node function will receive a state as the input
# In a LangGraph's MessageGraph, the state is simply a sequence of messages
def reflection_node(state: Sequence[BaseMessage]):
  # The node runs and invokes the reflection chain with the state messages as the input
  # The state is going to hold all of the critiques and previous generations that we had so far
  #
  # Remember the MessagesPlaceholder, now we're plugging in messages which is simply the state which is the agent's message history
  res = reflection_chain.invoke({"messages": state})

  # Once we get the response back from the LLM, we change it into a HumanMessage (instead of the default AIMessage)
  # We do this because we want to trick the LLM that the human is sending the critique
  # so it seems like we're having a conversation between the human and the assistant
  # This is a very important technique when we implement things with LangGraph
  #
  # and then we return the value and append it to the state (which happens under the hood)
  return HumanMessage(content=res.content)

#########
# GRAPH #
#########

# Initialize Graph
builder = MessageGraph()

# Add Generate Node
builder.add_node(GENERATE, generation_node)

# Add Reflection Node
builder.add_node(REFLECT, reflection_node)

# Set the starting node to be the generation node
builder.set_entry_point(GENERATE)

# This is a conditional edge function that takes the state and returns the key of the next node to execute
def should_continue(state: List[BaseMessage]):
  if len(state) > 6:
    return END
  return REFLECT

# Add conditional edge to the generation node
# add_conditional_edges is a routing function that tells LangGraph which node to execute next based on the return value of the should_continue function
# The 3rd argument is a path-mapping dictionary which maps the strings to the actual nodes in the graph
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
with open("old reflection graph.png", "wb") as f:
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