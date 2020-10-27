import streamlit as st
import streamlit.components.v1 as components
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


pages = ["About", "EDA", "The Model", "Conclusions"]
#tags = ["Target1", "Target2", "Target3", "Target4"]

st.sidebar.markdown("__Drug Viability ML Model Dashboard__")
page = st.sidebar.radio("Which section would you like to go to", options=pages)
st.sidebar.markdown('---')
st.sidebar.write('Created by Ryan Lewis')

st.title(page)
if page == "About":
    st.write('Blah blah blah here is some text for the body, put in the description from the github repo here')
    st.write('Figure out how to put a background image here and in the sidebar. At the bare minimum just change the colors to grey/light blue')

if page == "EDA":
    st.write("see if it's possible to ditch the top and right spines like in EDA.ipynb")
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
            orange_patch = mpatches.Patch(color='orange', label='Range Present in\n' + selection[0] + ' drugs')
            blue_patch = mpatches.Patch(color='blue', label='Range Absent in\n' + selection[0] + ' drugs')
            plt.legend(handles=[blue_patch, orange_patch], fontsize=8)
            #fig = plt.show()
            st.pyplot(fig)

if page == "The Model":
	st.write('Use one of the more recent models from the Acer...')
	graph1 = open('figures/Plotly.html', 'r', encoding='utf-8')
	source_code= graph1.read()
	print(source_code)
	components.html(source_code, height = 600, width = 800)
	#st.markdown(graph1_html, unsafe_allow_html=True)
	st.write("Brian said that putting in a grouped histogram comparing the the results for 2 or 3 of the models' predictions on a given MOA would be a better comparison")
	st.write('Maybe there shoudl be another two grouped histograms, one which compares precision and one which compares recall')
	st.write('Ask Anterra if there should be an explanation here for the usefulness of the precision/recall values: this is a drug development screening tool that would be used pre-preclinical trials when sending batches of drugs to be tested in vitro on human cells')

if page == "Conclusions":
    st.write('Blah blah blah here is some text for the body, put in the description from the github repo here')