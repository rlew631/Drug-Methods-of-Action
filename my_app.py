import streamlit as st
import streamlit.components.v1 as components
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

pages = ["Project Overview", "Exploratory Data Analysis", "Model Results", "Conclusions"]
#tags = ["Target1", "Target2", "Target3", "Target4"]

st.sidebar.markdown("__Drug Viability ML Model Dashboard__")
page = st.sidebar.radio("Which section would you like to go to", options=pages)
st.sidebar.markdown('---')
st.sidebar.write('Created by Ryan Lewis')
st.sidebar.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#e8f6fc,#b3e6ff);
    color: black;
}
</style>
""",
    unsafe_allow_html=True,
)

page_bg_img = '''
<style>
body {
background-image: url("https://raw.githubusercontent.com/rlew631/Drug-Methods-of-Action/main/Streamlit_BG.png");
background-size: cover;
}
</style>
'''

st.title(page)
if page == "Project Overview":
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown("The purpose of this project is to find a relationship between drug attributes and their corresponding methods of action.")
    st.markdown("The csv files used in this project can be found in the [Kaggle LISH-MOA Data Repository](https://www.kaggle.com/c/lish-moa/data)")
    st.markdown("Each drug experiment contains roughly 600 features related to genetic expression and 200 related to chemical attributes. A new model is made for each method of action and predicts the results as a function of how likely a given drug is to express an MOA. This is an incredibly useful tool which could allow researchers to predict a drug's viability and MOAs. This could improve the efficiency of the drug development process by selectively eliminating drug candidates which are unlikely to be viable before moving on to the screening and preclinical trial phases of development.")
if page == "Exploratory Data Analysis":
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.write('should i do a side by side comparison of the models if more than two are selected? I could do matplotlib subplots')
    train_targets = pd.read_csv('csvs/train_targets_scored.csv')
    train_features = pd.read_csv('csvs/train_features.csv')
    moas = train_targets.columns.values
    selection = st.multiselect("Select the MOA that you would like to view the feature profiles for", moas)
    if len(selection) > 0:
        st.write('You have selected: ' + str(selection))
        if len(selection) > 1:
        	st.markdown("__The dashboard only supports viewing the drug characteristics for one MOA at this time__")

        selected_drugs = train_targets[selection].loc[train_targets[selection[0]] == 1].index
        feat_slice = train_features.iloc[selected_drugs]
        values = train_features['g-0']
    
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # disable the warning message

        featurelist = ['g-0','g-1','g-2','c-0','c-1','c-2']
        plt.rcParams.update({"figure.facecolor":  (0.0, 0.0, 0.0, 0.0)})
        for featurename in featurelist:
            bincount = list(np.linspace(min(train_features[featurename]), max(train_features[featurename]), 60))
            N, bins, patches = plt.hist(values, bins = bincount, rwidth=0.75)
            for i in range(len(bincount)-1):
                for drug in feat_slice[featurename]:
                    if (bincount[i] <= drug <= bincount[i+1]):
                        patches[i].set_facecolor('orange')
            fig = plt.plot()
            if len(selection) > 1:
                plt.title('Marker ' + "'" + featurename + "'" + ': Value Counts for ' + selection[0] + ' Drugs')
            else:
            	plt.title('Marker ' + "'" + featurename + "'" + ': Value Counts', fontsize=22)
            plt.xlabel('Genetic/Chemical Marker Z-Score for All Drugs')
            plt.ylabel('Number of Values in Range')
            #ax.spines["top"].set_visible(False)  
            #ax.spines["right"].set_visible(False)
            #spines.Spine["top"].set_visible(False)  
            orange_patch = mpatches.Patch(color='orange', label='Range Present in\n' + selection[0] + '\ndrugs')
            blue_patch = mpatches.Patch(color='blue', label='Range Absent in\n' + selection[0] + '\ndrugs')
            plt.legend(handles=[blue_patch, orange_patch], fontsize=8)
            #fig = plt.show()
            sns.despine(top=True, right=True, left=False, bottom=False)
            st.pyplot(fig)

if page == "Model Results":
    st.markdown(page_bg_img, unsafe_allow_html=True)
    graph1 = open('figures/Plotly_hbarchart_2020-10-27.html', 'r', encoding='utf-8')
    source_code1= graph1.read()
    graph2 = open('figures/Plotly_all_logloss_2020-10-27.html', 'r', encoding='utf-8')
    source_code2= graph2.read()
    components.html(source_code1, height = 600, width = 800)
    st.write("Should the legend be put in the graph even if I can't make the box opaque? (might cover some of the points)")
    components.html(source_code2, height = 600, width = 800)
    #st.markdown(graph1_html, unsafe_allow_html=True)
    st.write('Maybe there should be another two grouped histograms,... one which compares precision and one which compares recall')
    st.write('Ask Anterra if there should be an explanation here for the usefulness of the precision/recall values: i.e. this is a drug development screening tool that would be used pre-preclinical trials when sending batches of drugs to be tested blah blah blah')

if page == "Conclusions":
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.write('It was surprising that the prediction scores between the linear model, ')