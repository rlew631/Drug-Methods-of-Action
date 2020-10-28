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
page = st.sidebar.radio("Select the Section you would like to view", options=pages)
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
""",unsafe_allow_html=True,)

page_bg_img = '''
<style>
body {
background-image: url("https://raw.githubusercontent.com/rlew631/Drug-Methods-of-Action/main/Streamlit_BG.png");
background-size: cover;
}
table {
  border-collapse: collapse;
  width: 100%;
  background-color: rgba(255,255,255,1);
}

th {
  border: 1px solid #dddddd;
  text-align: left;
  padding: 8px;
}
div.transbox {
  padding-top: 30px;
  padding-right: 20px;
  padding-bottom: 50px;
  padding-left: 20px;
  background-color: rgb(154, 186, 211);
  opacity: 0.8;
  width: 100%;
}
tr {
  background-color: rgba(154, 186, 211, 0.8);
}
</style>
'''

st.title(page)
if page == "Project Overview":
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown("""
    	<div class="transbox">
    <p>The purpose of this project is to find a relationship between drug attributes and their corresponding methods of action.
    	This could improve the efficiency of the drug development process by selectively eliminating drug candidates which are likely to
    	have additional unwanted MOAs (mechanisms of action) before moving on to the screening and preclinical trial phases of development.
    </p>
    <p>Each drug experiment contains roughly 600 features related to genetic expression and 200 related to chemical attributes.
    	A new model is made for each method of action and predicts the results as a function of how likely a given drug is to express an MOA.
    	This is an incredibly useful tool which could allow researchers to predict a drug's viability and MOAs.</p>
    <p>Each drug experiment contains roughly 600 features related to genetic expression and 200 related to chemical attributes.
    	A new model is made for each method of action and predicts the results as a function of how likely a given drug is to express an MOA.
    	This is an incredibly useful tool which could allow researchers to predict a drug's viability and MOAs. </p>
  </div>""", unsafe_allow_html=True)
    # st.markdown("""The purpose of this project is to find a relationship between drug attributes and their corresponding methods of action.
    # 	This could improve the efficiency of the drug development process by selectively eliminating drug candidates which are likely to
    # 	have additional unwanted MOAs (mechanisms of action) before moving on to the screening and preclinical trial phases of development.""")
    # st.markdown("""Each drug experiment contains roughly 600 features related to genetic expression and 200 related to chemical attributes.
    # 	A new model is made for each method of action and predicts the results as a function of how likely a given drug is to express an MOA.
    # 	This is an incredibly useful tool which could allow researchers to predict a drug's viability and MOAs.""")
    # st.markdown("The csv files used in this project can be found in the [Kaggle LISH-MOA Data Repository](https://www.kaggle.com/c/lish-moa/data)")
if page == "Exploratory Data Analysis":
    st.markdown(page_bg_img, unsafe_allow_html=True)
    #st.write('should i do a side by side comparison of the models if more than two are selected? I could do matplotlib subplots')
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
        #plt.rcParams.update({"figure.facecolor":  (0.0, 0.0, 0.0, 0.0)})
        sns.set(rc={'axes.facecolor':(0.0, 0.0, 0.0, 0.0),
        	'figure.facecolor':(0.0, 0.0, 0.0, 0.0),
        	'axes.grid' : False
        	})
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
            orange_patch = mpatches.Patch(color='orange', label='Range Present in\n' + selection[0] + '\ndrugs')
            blue_patch = mpatches.Patch(color='blue', label='Range Absent in\n' + selection[0] + '\ndrugs')
            plt.legend(handles=[blue_patch, orange_patch], fontsize=8, facecolor='#9abad3', framealpha=0.8)
            sns.despine(top=True, right=True, left=False, bottom=False)
            st.pyplot(fig)

if page == "Model Results":
    st.markdown(page_bg_img, unsafe_allow_html=True)
    graph1 = open('figures/Plotly_hbarchart_2020-10-27.html', 'r', encoding='utf-8')
    source_code1= graph1.read()
    graph2 = open('figures/Plotly_all_logloss_2020-10-27.html', 'r', encoding='utf-8')
    source_code2= graph2.read()
    components.html(source_code1, height = 600, width = 800)
    components.html(source_code2, height = 600, width = 800)
    #st.markdown(graph1_html, unsafe_allow_html=True)
if page == "Conclusions":
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown("""<div class="transbox">
    <p>The results from this modelling phase look promising for further development.
 It is worth noting that several other models which can be found on Kaggle were
 able to achieve a higher accuracy using neural networks.</p>
    <p>Based on the highly similar log loss values from the logistic regression,
 random forest regression and linSVR models it is unlikely that further improvements
 could be made to the model using stacked ensembling or a naive bayes model. Knowing
 that there are improvements which can be made to the prediction model the Linear SVR model
 is the clear winner with a runtime of only 34s and performance comparable to the
 logistic regression and random forest models.</p>
 <table style="width:100%">
  <tr>
    <th>Model</th>
    <th>Runtime</th>
    <th>Log Loss Value</th>
  </tr>
  <tr>
    <td>Logistic Regression</td>
    <td>32 minutes 9 seconds</td>
    <td>0.02006</td>
  </tr>
  <tr>
    <td>Random Forest Regression</td>
    <td>26 minutes 30 seconds</td>
    <td>0.02031</td>
  </tr>
  <tr>
    <td>Linear SVR</td>
    <td>0 minutes 34 seconds</td>
    <td>0.02059</td>
  </tr>
  <tr>
    <td>NuSVR</td>
    <td>59 minutes 27 seconds</td>
    <td>0.07413</td>
  </tr>
</table>
    <p><p>Knowing there was an increase in performance seen by the NN models used on kaggle it is likely that
there are gains to be had by including interaction terms in the next modelling process. The zero-centered
features would suit this process well and likely provide some insight into the chemical and genetic markers
which need to be present simultaneously if used with a logistic regression model.</p>
     <p>On a related note, several downsampling methods were attempted with the 4 models in order to reduce the inherent
bias present in the highly class-imbalanced feature data. TomekLinks was the only method which showed any
improvement, but the improvement was not greater than the variance between runs (when running without a random
state seed value for the models) and was deemed not worth the extra complexity and run time it added to the model
training process. In the next project iteration it would be worth looking into several upsampling methods to see
if there's room to improve the model's prediction score.</p></div>""", unsafe_allow_html=True)



