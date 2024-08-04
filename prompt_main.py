import os
from constants import openai_key #set your api key in openai_key variable in constants file
from langchain_openai import OpenAI
from langchain import PromptTemplate # Prompt llm
from langchain.chains import LLMChain # LLMchain 
from langchain.chains import SimpleSequentialChain #For multiple chian templates,  but only shows the last output
from langchain.chains import SequentialChain #For seeing all the chains' result at once
from langchain.memory import ConversationBufferMemory #For printing all the conversational details on the page
import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

# Streamlit framework
st.title('Tourism Plan')
input_text=st.text_input("Search the place you want to travel")

## OpenAI LLMS
llm=OpenAI(temperature=0.8)  ##how much control the agent will have on the output, range 0-1,  more number,  more control

# Prompt Template-1
first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Great Choice !!! \nHere is brief about the place {name}"
)
# Assigning conversation memory
location_memory=ConversationBufferMemory(input_key='name',memory_key='chat_history')

## LLMChain model
chain=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='Location',memory=location_memory)

# Prompt Template-2
second_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="These are Top 10 sight-seeing places for {name}"
)

# Assigning conversation memory
top10_memory=ConversationBufferMemory(input_key='name',memory_key='description_history')

## Another LLMChain model
chain1=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='top10',memory=top10_memory)


# Prompt Template-3
third_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Below is a tabular structure of famous local cusine of {name}, differentiate by veg or non-veg,  along with their ingredients."
)

# Assigning conversation memory
food_memory=ConversationBufferMemory(input_key='name',memory_key='description_history')

## Another LLMChain model
chain2=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='local_cusine',memory=food_memory)



# Prompt Template-4
fourth_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Below is a tabular structure of cost for transporation methods, accomodation, and sightseeing to {name} from Bangalore"
)

# Assigning conversation memory
expense_memory=ConversationBufferMemory(input_key='name',memory_key='description_history')

## Another LLMChain model
chain3=LLMChain(llm=llm,prompt=fourth_input_prompt,verbose=True,output_key='expense',memory=expense_memory)


parent_chain=SimpleSequentialChain(chains=[chain,chain1],verbose=True)
all_chain=SequentialChain(chains=[chain,chain1,chain2,chain3],input_variables=['name'],output_variables=['Location','top10','local_cusine','expense'],verbose=True)

if input_text:
    st.write(all_chain({'name':input_text}))  

    with st.expander('Location Name'):
        st.info(location_memory.buffer)
    
    with st.expander('Top 10 Sightseeing'):
        st.info(top10_memory.buffer)
    
    with st.expander('Local Cusines'):
        st.info(food_memory.buffer)
    
    with st.expander('Approx Expenses from Bangalore'):
        st.info(expense_memory.buffer)
    
#input_text1=st.text_input("Please advice what more information do you want.") # Try to take response and regenrate until user stops