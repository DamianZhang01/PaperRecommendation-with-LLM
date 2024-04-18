import streamlit as st
import numpy as np
import pandas as pd
import os
import h5py
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from database import setup_database, get_paper_by_id, get_paper_by_title, Paper
from vecDB import VecDB
from chatConnect import GPTChat
from gpt_embedding import gpt_embedding

# load the word2vec model
word_vectors = KeyedVectors.load_word2vec_format('/Users/zyw/Documents/SI618/SI_618_WN_24_Files/data/GoogleNews-vectors-negative300-SLIM.bin', binary=True) 

# initialize the sqlite db
session = setup_database()

# initialize the vecDB
db_microsoft = VecDB(emb_dim=300, filepath='microsoft_db.h5')
db_graphsage = VecDB(emb_dim=128, filepath='graphsage_db.h5')
db_gpt = VecDB(emb_dim=768, filepath='gpt_db.h5')


# def function to retrive the embeddings from query string
# functions to embed the text into a vector
def document_vector(text, word_vectors):
    words = text.split()
    word_vecs = [word_vectors[word] for word in words if word in word_vectors.key_to_index]
    return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(word_vectors.vector_size)

def get_embeddings(query, model):
    if model == 'Microsoft':
        return document_vector(query, word_vectors)
    elif model == 'GraphSage':
        return db_graphsage.get(index=0)
    elif model == 'GPT':
        return gpt_embedding(query)


# initialize the session state
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home_page'

# home page
def render_home():
    # title
    st.title('Paper Recommendation System')
    # introduction
    st.write('Welcome to the paper recommendation system!')
    st.write('This system is designed to help you find papers that are similar to the one you are interested in.')
    st.write('Here\'s some functions you could use:')

    # list of functions
    # 3 columns
    col1, col2, col3 = st.columns(3)

    # create a button in each column
    with col1:
        if st.button('Paper Search'):
            st.session_state['current_page'] = 'func_search'
    with col2:
        if st.button('Chatbot'):
            st.session_state['current_page'] = 'func_chatbot'
    with col3:
        if st.button("Paper Recommendation"):
            st.session_state['current_page'] = 'func_recommendation'


# paper search page
def render_page_search():
    st.title("Paper Search")
    st.write('Enter the title of the paper you are interested in.')

    # input interface
    # input: paper title
    user_input = st.text_input("Please type in a paper title:")
    # submit button
    if st.button('Search Paper'):
        # get paper
        input_paper = get_paper_by_title(session, user_input)
        if input_paper:
            st.write(f'Here are the details of the paper:')
            st.markdown(f"""
                        **Title:** {input_paper['title']} <br>
                        **Authors:** {input_paper['authors']} <br>
                        **Year:** {input_paper['year']} <br>
                        **Journal:** {input_paper['venue']} <br>
                        **Abstract:** {input_paper['abstract']}
                    """, unsafe_allow_html=True)

    # search function
    # search_result = search_paper(user_input)

    # display search result
    # st.write(search_result)

    if st.button("back to home page"):
        st.session_state['current_page'] = 'home_page'


# chatbot page
def render_page_chatbot():
    st.title("Talk with gpt3.5 about your idea")
    st.write('You could chat with the chatbot here.')

    gpt_chat = GPTChat("tell me what you know about this paper")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask your question?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = gpt_chat.get_reply(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # chatbot
    # chatbot = load_chatbot()
    # chatbot()

    if st.button("back to home page"):
        st.session_state['current_page'] = 'home_page'


# paper recommendation page
def render_page_recommendation():
    st.title("Paper Recommendation") 
    st.write('Enter the title or query that you are interested in, and we will recommend some papers for you.')

    # input interface 
    # options for input mode
    model_selection = st.selectbox('Select the input mode:', ['query', 'title'])
    # options for model selection
    model_selection = st.selectbox('Select a model:', ['Microsoft', 'GraphSage', 'GPT', 'Ensemble'])
    # options for recommendation number
    recommendation_number = st.selectbox('Select the number of recommendations:', [3, 5, 10])
    # inout: paper title
    output = []
    if model_selection == 'title':
        user_input_title = st.text_input("Please type in a paper title:")

        if st.button('recommend papers'):
            # get paper
            input_paper = get_paper_by_title(session, user_input_title)
            # recommend
            if input_paper:
                id = input_paper['id']
                if model_selection == 'Microsoft':
                    output = db_microsoft.most(
                        db_microsoft.get(index=id),
                        func=cosine_similarity,
                        n=recommendation_number,
                        desc=True
                    )
                elif model_selection == 'GraphSage':
                    output = db_graphsage.most(
                        db_graphsage.get(index=id),
                        func=cosine_similarity,
                        n=recommendation_number,
                        desc=True
                    )
                elif model_selection == 'GPT':
                    output = db_gpt.most(
                        db_gpt.get(index=id),
                        func=cosine_similarity,
                        n=recommendation_number,
                        desc=True
                    )
                
                # query in papers.db to get detailed information
                detailed_outputs = []
                for (paper_id, score) in output:
                    paper_details = get_paper_by_id(session, paper_id)
                    if paper_details:
                        paper_info = {
                            "Title": paper_details['title'],
                            "Authors": paper_details['authors'],
                            "Year": paper_details['year'],
                            "Score": f"{score:.2f}"
                        }
                        detailed_outputs.append(paper_info)
                    else:
                        st.write(f"Paper ID {paper_id} not found in database.")

                # show detail information about each paper
                st.write(f'Ok, Here\'s some recommendations for you: ')
                paper_df = pd.DataFrame(detailed_outputs)
                st.table(paper_df)



    else:
        user_input_query = st.text_input("Please type in a query:")
        
        # submit button
        if st.button('recommend papers'):
            if model_selection == 'Microsoft':
                output = db_microsoft.most(
                    get_embeddings(user_input_query, model_selection),
                    func=cosine_similarity,
                    n=recommendation_number,
                    desc=True
                )
            elif model_selection == 'GraphSage':
                output = db_graphsage.most(
                    get_embeddings(user_input_query, model_selection),
                    func=cosine_similarity,
                    n=recommendation_number,
                    desc=True
                )
            elif model_selection == 'GPT':
                output = db_gpt.most(
                    get_embeddings(user_input_query, model_selection),
                    func=cosine_similarity,
                    n=recommendation_number,
                    desc=True
                )
            # st.write(output)
            # query in papers.db to get detailed information
            detailed_outputs = []
            for (paper_id, score) in output:
                paper_details = get_paper_by_id(session, paper_id)
                if paper_details:
                    paper_info = {
                        "Title": paper_details['title'],
                        "Authors": paper_details['authors'],
                        "Year": paper_details['year'],
                        "Score": f"{score:.2f}"
                    }
                    detailed_outputs.append(paper_info)
                else:
                    st.write(f"Paper ID {paper_id} not found in database.")

            # show detail information about each paper
            st.write(f'Here are the paper recommendations for you: ')
            paper_df = pd.DataFrame(detailed_outputs)
            paper_df.index = paper_df.index + 1
            st.table(paper_df)

    if st.button("back to home page"):
        st.session_state['current_page'] = 'home_page'



# according to the current page, render different pages
if st.session_state['current_page'] == 'home_page':
    render_home()
elif st.session_state['current_page'] == 'func_search':
    render_page_search()
elif st.session_state['current_page'] == 'func_chatbot':
    render_page_chatbot()
elif st.session_state['current_page'] == 'func_recommendation':
    render_page_recommendation()

