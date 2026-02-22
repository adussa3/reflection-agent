# The chains file will hold all of the prompts and chains that we're going to be usign in our LangGraph graph

# The ChatPromptTemplate is a class that holds our content that we send to the LLM as a human message,
# or that we receive back from the LLM as an answer that's tagged as an assistant message
# 
# The MessagesPlaceholder is a class that gives us the flexability to put a placeholder for future messages that we're going to get
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

###########
# PROMPTS #
###########

# The generation prompt is going to generate the tweets which will be revised over and over again after the feedback we get from the reflection prompt
# It's going to revise the tweet until it gets the perfect tweet
generation_prompt = ChatPromptTemplate.from_messages([
  (
    "system",
    "You are a twitter techie influencer assistant tasked with writing excellet twitter posts."
    " Generate the best twitter post possible for the user's request."
    " If the user provides critique, respond with a revised version of your previous attempts."
  ),
  # We want to put a placeholder to hold all of the reflections and revisions that we had earlier
  #
  # When we initialize the generation prompt, we plug into the prompt the messages of our history
  MessagesPlaceholder(variable_name="messages")
])

# The reflection prompt will act as our critique and review the output and critisize it and give us feedback on how to improve it
reflection_prompt = ChatPromptTemplate.from_messages([
  (
    "system",
    "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
    " Always provide detailed recommendations, including requests for length, virality, style, etc."
  ),
  # We want to put here a placeholder for other messages. These will be the "history" messages that our agent is going to invoke
  # and to critisize and to get recommendations over and over again
  #
  # When we initialize the reflection prompt, we plug into the prompt the messages of our history
  MessagesPlaceholder(variable_name="messages")
])

# To create chains, we need to initialize an LLM
# By default, the ChatOpenAI initialization will use the gpt-3.5-turbo model
# By default, ChatOpenAI will look for the "OPENAI_API_KEY" environment variable to authenticate with the OpenAI API
llm = ChatOpenAI()

##########
# CHAINS #
##########

# We "pipe" the generation prompt and reflection prompt into the LLM
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm