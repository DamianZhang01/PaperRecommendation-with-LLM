import streamlit as st

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
        if st.button('paper search'):
            st.session_state['current_page'] = 'func_search'
    with col2:
        if st.button('chatbot'):
            st.session_state['current_page'] = 'func_chatbot'
    with col3:
        if st.button("paper recommendation"):
            st.session_state['current_page'] = 'func_recommendation'


# paper search page
def render_page_search():
    st.title("paper search function")
    st.write('You could type in the title of the paper you are interested in, and we will search for you.')

    # input interface
    # input: paper title
    user_input = st.text_input("Please type in a paper title:")
    # submit button
    if st.button('search paper'):
        st.write(f'Ok, Here\'s the search result for you: {user_input}!')

    # search function
    # search_result = search_paper(user_input)

    # display search result
    # st.write(search_result)

    if st.button("back to home page"):
        st.session_state['current_page'] = 'home_page'


# chatbot page
def render_page_chatbot():
    st.title("Chatbot function")
    st.write('You could chat with the chatbot here.')

    # chatbot
    # chatbot = load_chatbot()
    # chatbot()

    if st.button("back to home page"):
        st.session_state['current_page'] = 'home_page'


# paper recommendation page
def render_page_recommendation():
    st.title("paper recommendation function") 
    st.write('You could type in the title of the paper you are interested in, and we will recommend some papers for you.')

    # input interface 
    # inout: paper title
    user_input = st.text_input("Please type in a paper title:")
    # options for model selection
    model_selection = st.selectbox('Select a model:', ['Microsoft', 'GraphSage', 'Bert', 'Ensemble'])
    # options for recommendation number
    recommendation_number = st.selectbox('Select the number of recommendations:', [3, 5, 10])

    # submit button
    if st.button('recommend papers'):
        st.write(f'Ok, Here\'s some recommendations for you: {user_input}!')

    # models
    # model = load_model()
    # recommendations = model(user_input)

    # display recommendations
    # st.write(recommendations)

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

