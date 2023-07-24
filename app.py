import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template, styl
from streamlit_chat import message


def get_pdf_text(pdf_docs):
    text = ""
    
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        #looping through each page to get text from each page in the pdf
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(raw_text):

    text_splitter = CharacterTextSplitter(chunk_size=1000,
                                          chunk_overlap=200,
                                          separator="\n",
                                          length_function=len)
    
    chunks = text_splitter.split_text(raw_text)

    return chunks

def get_vectorstore(chunks):
    
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    
    return vectorstore
    

def get_conversation_chain(vectorstore):
    
    llm = OpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs= {"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                               retriever=vectorstore.as_retriever(),
                                                               memory=memory)
    
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):    #just change the varaible name to something else from
        if i % 2 == 0 :                                            #from 'message' to messag, if you are using streamlit-chat
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            #message(messag.content, is_user = True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            #message(messag.content)
def main():
    load_dotenv()
    
    #setting the page configuration such as page title and other things
    st.set_page_config(page_title='chat with your pdfs', page_icon=':books:')
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    #to add a header
    st.header("Start Chatting with your PDFs")

    st.write(css, unsafe_allow_html=True)

    #adding the text input
    user_input = st.text_input("Ask your Question here")
    st.markdown(styl, unsafe_allow_html=True)
    
    if user_input:
        handle_user_input(user_input)

    #now to add a sidebar
    with st.sidebar:
        #writing a subheader
        st.subheader("Your Files : ")
        
        #now to upload the files
        pdf_docs = st.file_uploader("Upload your PDF files to Start Chatting", accept_multiple_files=True)
        
        #then adding a button
        if st.button("Process"):
            
            #to show a spinner
            with st.spinner("processing"):
            
                #to load our files or text
                raw_text = get_pdf_text(pdf_docs)

                #now to get the chunks of the text
                chunks = get_text_chunks(raw_text)

                #now to make vector representations of our chunks and create a vectorstore or database
                vectorstore = get_vectorstore(chunks)

                #now to make a conversation chain and add some memory to it
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()




