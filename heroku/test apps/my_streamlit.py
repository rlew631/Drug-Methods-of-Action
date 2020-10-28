  
#import package
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle

#preprocessing
import spacy
import re
import string

#import the data
image = Image.open("figures/background.jpg")
#intro
#st.sidebar.write("Thank you for visiting Ryan's <b>MOA prediction model</b> site")
st.sidebar.markdown("__Ryan's MOA Prediction Model dashboard__")

html = """
  <style>
    /* Disable overlay (fullscreen mode) buttons */
    
    .overlayBtn {
      display: none;
    }

    /* 2nd thumbnail */
    .element-container:nth-child(4) {
      top: -266px;
      left: 350px;
    }

    /* 1st button */
    .element-container:nth-child(3) {
      left: 10px;
      top: -60px;
    }

    /* 2nd button */
    .element-container:nth-child(5) {
      left: 360px;
      top: -326px;
    }
  </style>
"""
st.sidebar.markdown(html, unsafe_allow_html=True)

#img1 = st.sidebar.image("https://www.w3schools.com/howto/img_forest.jpg", width=300)
img1 = st.sidebar.image("https://www.w3schools.com/howto/img_forest.jpg", use_column_width=True)
st.sidebar.button("Show", key=1)

#img2 = st.sidebar.image("https://www.w3schools.com/howto/img_forest.jpg", width=300)
img2 = st.sidebar.image("https://www.w3schools.com/howto/img_forest.jpg", use_column_width=True)
st.sidebar.button("Show", key=2)




Title_html = """
    <style>
        .title h1{
          user-select: none;
          font-size: 43px;
          color: white;
          background: repeating-linear-gradient(-45deg, red 0%, yellow 7.14%, rgb(0,255,0) 14.28%, rgb(0,255,255) 21.4%, cyan 28.56%, blue 35.7%, magenta 42.84%, red 50%);
          background-size: 600vw 600vw;
          -webkit-text-fill-color: transparent;
          -webkit-background-clip: text;
          animation: slide 10s linear infinite forwards;
        }
        @keyframes slide {
          0%{
            background-position-x: 0%;
          }
          100%{
            background-position-x: 600vw;
          }
        }
        .reportview-container .main .block-container{
            padding-top: 3em;
        }
        body {
            background-image:url('https://images2.alphacoders.com/692/692539.jpg');
            background-position-y: -200px;
        }
        @media (max-width: 1800px) {
            body {
                background-position-x: -500px;
            }
        }
        .Widget.stTextArea, .Widget.stTextArea textarea {
        height: 586px;
        width: 400px;
        }
        h1{
            color: #28784D
        }
        .sidebar-content {
            width:25rem !important;
        }
        .sidebar.--collapsed .sidebar-content {
         margin-left: -25rem;
        }
    </style> 
    
    <div>
        <h1>Welcome to the Myers Briggs Prediction App!</h1>
    </div>
    """
st.markdown(Title_html, unsafe_allow_html=True) #Title rendering

# Calculate our prediction
# import models
EI = pd.read_pickle('pickled_models/EI_Logistic Reg.pkl')
NS = pd.read_pickle('pickled_models/NS_Logistic Reg.pkl')
FT = pd.read_pickle('pickled_models/FT_Logistic Reg.pkl')
PJ = pd.read_pickle('pickled_models/PJ_Logistic Reg.pkl')

# import transformations
tfidf = pd.read_pickle('pickled_transformations/tfidf.pkl')
TopicModel = pd.read_pickle('pickled_transformations/NMF.pkl')

# Create our list of punctuation marks
punctuations = string.punctuation
# Load English tokenizer, tagger, parser, NER and word vectors
parser = spacy.load('en_core_web_sm')
# Create our list of stopwords
stop_words = spacy.lang.en.stop_words.STOP_WORDS
# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)
    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    # return preprocessed list of tokens
    return ' '.join(mytokens)
alphanumeric = lambda x: re.sub('\w*\d\w*', '', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x.lower())

new_text = pd.Series(text).apply(spacy_tokenizer).map(alphanumeric).map(punc_lower)
X_test_tfidf = tfidf.transform(pd.Series(new_text))

def display_topics(model, feature_names, no_top_words, topic_names=None):
    """
    Takes in model and feature names and outputs 
    a list of string of the top words from each topic.
    """
    topics = []
    for ix, topic in enumerate(model.components_):
        topics.append(str(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])))
    return topics

topics = display_topics(TopicModel, tfidf.get_feature_names(), 15)
topic_word = pd.DataFrame(TopicModel.components_.round(3),
             index =  topics,
             columns = tfidf.get_feature_names())

X_test_topic_array = TopicModel.transform(pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf.get_feature_names()))

X_test_topics = pd.DataFrame(X_test_topic_array.round(5),
             columns = topics)

pred_list = []
if EI.predict(X_test_topics) == 1:
    pred_list.append('E')
else:
    pred_list.append('I')
if NS.predict(X_test_topics) == 1:
    pred_list.append('N')
else:
    pred_list.append('S')
if FT.predict(X_test_topics) == 1:
    pred_list.append('F')
else:
    pred_list.append('T')
if PJ.predict(X_test_topics) == 1:
    pred_list.append('P')
else:
    pred_list.append('J')
prediction = ''.join(pred_list)

if text == '':
    st.image(image, use_column_width=True)
    st.markdown("<div class='title'><h1>Start by writing text on the left sidebar</h1></div>", unsafe_allow_html=True)
if text != '':
    st.header('We guess that you are:')
    predict_html = f"<div class='title'><h1>{prediction}</h1></div>"
    st.markdown(predict_html, unsafe_allow_html=True)
    st.image(f'images/{prediction}.png',width=340)
    st.header('Are we correct?')
    st.write('Find more information here:')
    st.write(f'https://www.16personalities.com/{prediction.lower()}-personality')

if st.checkbox('Generate Word Cloud') == 1:
    try:
        # Generate WordCloud
        from wordcloud import WordCloud 

        # Generate a word cloud image
        wordcloud = WordCloud(width = 1000, height = 1000,
                        background_color ='white',
                        min_font_size = 20).generate(text)
        st.header('Let analyze your text:')
        st.write(text)
        st.header('Word cloud of your text:')
        # Display the generated image:
        # the matplotlib way:
        fig = plt.figure(figsize=(10,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        st.pyplot(fig)
    except ValueError:
        pass














# # set sliders to assign duration and campaign goal

# df['maxplayers'] = st.slider('max players', 1,20,1)
# df['age'] = st.number_input('recommended age',1,25,1)
# df['num_of_awards'] = st.slider('awards', 0,200,0)
# df['averageweight'] = st.slider('complexity 0-5',0.0,5.0,2.5,.1)
# df['podcasts'] = st.number_input('number of podcast videos',0,5000,0) 
# df['prime'] = st.checkbox('sold on amazon prime?')

# # List out the categorical variables

# mechanic = ['(MECHANIC) Cooperative Game',
#  '(MECHANIC) Dice Rolling',
#  '(MECHANIC) Hand Management',
#  '(MECHANIC) Modular Board',
#  '(MECHANIC) Player Elimination',
#  '(MECHANIC) Set Collection',
#  '(MECHANIC) Team-Based Game',
#  '(MECHANIC) Trading',
#  '(MECHANIC) Variable Player Powers']


# award = ['(AWARD)golden geek best board game expansion',
#  '(AWARD)golden geek best family board game',
#  '(AWARD)golden geek best party board game',
#  '(AWARD)gra roku game of the year',
#  '(AWARD)hra roku',
#  "(AWARD)japan boardgame prize voters' selection",
#  '(AWARD)jocul anului în românia best game in romanian',
#  '(AWARD)juego del año',
#  '(AWARD)mind-spielepreis',
#  '(AWARD)nederlandse spellenprijs',
#  '(AWARD)spiel des jahres']

# theme = ['(THEME) Action / Dexterity',
#  '(THEME) Card Game',
#  "(THEME) Children's Game",
#  '(THEME) City Building',
#  '(THEME) Expansion for Base-game',
#  '(THEME) Fantasy',
#  '(THEME) Fighting',
#  '(THEME) Medieval',
#  '(THEME) Territory Building',
#  '(THEME) Wargame']

# subgroup = ['(SUBGROUP) Crowdfunding: Kickstarter',
#  '(SUBGROUP) Players: Games with Solitaire Rules']

# domain = ['(DOMAIN) Family Games',
#  '(DOMAIN) Party Games',
#  '(DOMAIN) Strategy Games',
#  '(DOMAIN) Thematic Games']

# # create selection boxes for each categorical option

# m = st.multiselect('Mechanics in Boardgame', mechanic, format_func = lambda x: x[10:])
# a = st.multiselect('Awards', award, format_func = lambda x: x[7:])
# t = st.multiselect('Theme of game', theme, format_func = lambda x: x[7:])
# df['(SUBGROUP) Crowdfunding: Kickstarter'] = st.checkbox('Was it kickstarted?')
# '''
# If so, you should checkout my [other app](https://vast-beach-62642.herokuapp.com/)
# '''

# df['(SUBGROUP) Players: Games with Solitaire Rules'] = st.checkbox('Play by yourself?')
# d = st.multiselect('Category', domain, format_func = lambda x: x[8:])

# for i in m + a + t + d:
#     df[i] = 1

# test_array = np.array(df)


# st.dataframe(df)

# if prediction > 1000:
#     st.balloons()