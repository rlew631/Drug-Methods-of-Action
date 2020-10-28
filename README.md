# Drug-Methods-of-Action

## Project Overview

The purpose of this project is to find a relationship between drug attributes and their corresponding methods of action

The csv files used in this project can be found in the [Kaggle LISH-MOA Data Repository](https://www.kaggle.com/c/lish-moa/data)

Each drug experiment contains roughly 600 features related to genetic expression and 200 related to chemical attributes. A new model is made for each method of action and predicts the results as a function of how likely a given drug is to express an MOA. This is an incredibly useful tool which could allow researchers to predict a drug's viability and MOAs. This could improve the efficiency of the drug development process by selectively eliminating drug candidates which are unlikely to be viable before moving on to the screening and preclinical trial phases of development.

## Implementation

## Conclusion
The results from this modelling phase look promising for further development. It is worth noting that several other models which can be found on Kaggle were able to achieve a higher accuracy using neural networks.

Based on the highly similar log loss values from the logistic regression, random forest regression and linSVR models it is unlikely that further improvements could be made to the model using stacked ensembling or a naive bayes model. Knowing that there are improvements which can be made to the prediction model the Linear SVR model is the clear winner with a runtime of only 34s and performance comparable to the logistic regression and random forest models.

| Model | Runtime | Logloss Value |
|-----------------------|---------|----------------|
| Logistic Regression | 32 minutes 9 seconds | 0.02006 |
| Random Forest Regression | 26 minutes 30 seconds | 0.02031 |
| Linear SVR | 0 minutes 34 seconds | 0.02059 |
| NuSVR | 59 minutes 27 seconds | 0.07413 |

## Next Steps

Based on the increase in performance seen by the NN models used on kaggle it is likely that there are gains to be had by including interaction terms in the next modelling process. The zero-centered features would suit this process well and likely provide some insight into the chemical and genetic markers which need to be present simultaneously if used with a logistic regression model.

On a related note, several downsampling methods were attempted with the 4 models in order to reduce the inherent bias present in the highly class-imbalanced feature data. TomekLinks was the only method which showed any improvement, but the improvement was not greater than the variance between runs (when running without a random state seed value for the models) and was deemed not worth the extra complexity and run time it added to the model training process. In the next project iteration it would be worth looking into several upsampling methods to see if there's room to improve the model's prediction score. 
