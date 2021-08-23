# Machine-Learning_Credit-Risk-Analysis

### Overview
"No man's credit is as good as his money." That said, credit drives our economy and Fast Lending, a peer to peer lending services company, wants to use machine learning to predict credit risk for loans. Ideally, machine learning will help Fast Lending will improve accuracy, speed, and reliability of identifying good candidates for risk, ultimately reducing default rates. This analysis builds several machine learning models to predict and evaluate credit risk, including using techniques such as resampling and boosting. Loan data is inherently an unbalanced classication problem as the vast majority of loans paid as agreed outnumber loans in default or at risk of default.

Resources: Python 3.8.5 with Imbalanced-Learn, Scikit-Learn, and Pandas libraries.


### Results



##### Naive Random Model
![Naive_Random_Oversampling](https://github.com/ozloty06/Machine-Learning_Credit-Risk-Analysis/blob/main/Images/Naive_Random_Oversampling.png)

- The balanced accuracy score is 65.47%
- The high risk precision is only 1% while sensitivity is 72% with a very low F1 of 2%.
- The low risk precision is almost 100% while sensitivity is 59% with an F1 of 74%.
- It is likely the high population of the low risk group made precision very high.
- The F score for both high and low risk on this model seems low and our hope is other models would improve on this.


##### SMOTE Oversampling Model
![SMOTE_Oversampling](https://github.com/ozloty06/Machine-Learning_Credit-Risk-Analysis/blob/main/Images/SMOTE_Oversampling.png)

- The balanced accuracy score is 66.20%
- The high risk precision is only 1% while sensitivity is 63% with a very low F1 of 2%.
- The low risk precision is almost 100% while sensitivity is 69% with an F1 of 82%.
- The F score of the low-risk group is reasonably high.


##### Undersampling Model
![Undersampling](https://github.com/ozloty06/Machine-Learning_Credit-Risk-Analysis/blob/main/Images/Undersampling.png)

- The balanced accuracy score is 66.20%
- The high risk precision is only 1% while sensitivity is 69% with a very low F1 of only 1%.
- The low risk precision is almost 100% while sensitivity is 40% with an F1 of 57%.
- This model did not perform as well as the SMOTE Oversampling Model.


##### Combination Sampling Model
![Combination_Sampling](https://github.com/ozloty06/Machine-Learning_Credit-Risk-Analysis/blob/main/Images/Combination_Sampling.png)

- The balanced accuracy score is 54.47%
- The high risk precision is only 1% while sensitivity is 72% with a very low F1 of 2%.
- The low risk precision is almost 100% while sensitivity is 57% with an F1 of 73%.
- This model did not perform as well as the SMOTE Oversampling Model.


##### Balanced Random Forest Classifier Model
![Balanced_Random_Forest_Classifier](https://github.com/ozloty06/Machine-Learning_Credit-Risk-Analysis/blob/main/Images/Balanced_Random_Forest_Classifier.png)

- The balanced accuracy score is 78.85%
- The high risk precision is only 3% while sensitivity is 70% with a very low F1 of 6%.
- The low risk precision is almost 100% while sensitivity is 87% with an F1 of 93%.
- This model performed better than all of the prior models based on the higher balanced accuracy and F-scores.


##### Easy Ensemble AdaBoost Classifier Model
![Easy_Ensemble_AdaBoost_Classifier](https://github.com/ozloty06/Machine-Learning_Credit-Risk-Analysis/blob/main/Images/Easy_Ensemble_AdaBoost_Classifier.png)

- The balanced accuracy score is 93.16%
- The high risk precision is only 9% while sensitivity is 92% with a very low F1 of 16%.
- The low risk precision is almost 100% while sensitivity is 94% with an F1 of 97%.
- This model performed the best of all of the prior models based on the higher balanced accuracy and F-scores.


### Summary
Based on our analysis of loan credit risk data, the best machine learning model to use for assessing loan default risk is the Easy Ensemble AdaBoost Model due to the highest F-scores of 16% for high risk and 97% for low risk. 

That said, all of the models that were reviewed as part of this assessment showed a low precision. This may be a result of the models being overfit as our data was not reduced down to only the most essential columns nor was the data reviewed to ensure column data contained non duplicative information. It's possible that some of the data may be a reinterpretation of other data contained in the data set rather than representing distinct credit worthiness information.

The low precision means a lot of risk will be inaccurately considered high risk, leaving Fast Lending to possibly deny these applications and leave a great deal of loan opportunity behind. 

Based on the limitations seen across the models for precision, it is not recommended any of the models be used. 
