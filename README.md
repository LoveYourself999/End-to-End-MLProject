# End-to-End-MLproject

## 1. Framing the problem and looking at the big picture.
- Project Purpose: Predicting High-Risk COVID-19 Patient Rates in Canada
The primary objective of this project is to forecast the rate of high-risk COVID-19 patients in hospitals throughout Canada. High-risk patients are operationally defined as the ratio of patients in the Intensive Care Unit (ICU) to the total number of patients in the hospital.
- Methodology:
To achieve this goal, we employed COVID-19 vaccination and test data as essential training features. These data points form the basis for training three distinct machine learning algorithms: Linear Regression, Support Vector Machine (SVM), and Decision Tree. Finally, model performance is assessed by comparing their Mean Absolute Percentage Error (MAPE).
- This project can be seen as:
  - Supervised Learning: The training examples are labeled, facilitating the learning process.
  - Regression Task: The objective is to predict a continuous value.
  - Batch Learning: The project follows a batch learning approach, implying there is no continuous inflow of data into the system, and adjustments to changing data do not need to be made rapidly.
- Looking at the Big Picture:
The predictions derived from our models play a pivotal role in evaluating the ratio of high-risk patients within hospitals. This metric serves as a valuable tool for other regions, enabling them to predict potential ICU patients based on vaccination data. The insights gained can significantly contribute to governmental decision-making processes, particularly in managing ICU capacity and promoting strategic vaccine distribution. The broader implication of this project extends beyond immediate application, offering a framework for proactive healthcare planning in the context of the ongoing pandemic.
## 2. A description of the dataset and 3 graphs of EDAs.
Fetch data and create target column:
- Load the dataset from github: https://github.com/owid/covid-19-data/tree/master/public/data  
- Extract data of Canada from 2020-04-02 to 2022-01-01  
- Create a new column "high_risk" by dividing "icu_patients" by "hosp_patients"  

__Note that the data has been updated in real-time (The row number will be updated), so the results and graphs may vary depending on the dataset changes.__  
  
Description of the dataset:
- total_cases: Total confirmed cases of COVID-19. Counts can include probable cases, where reported.
- new_cases: New confirmed cases of COVID-19. Counts can include probable cases, where reported. In rare cases where our source reports a negative daily change due to a data correction, we set this metric to NA.
- new_cases_smoothed: New confirmed cases of COVID-19 (7-day smoothed). Counts can include probable cases, where reported.
- total_cases_per_million:Total confirmed cases of COVID-19 per 1,000,000 people. Counts can include probable cases, where reported.
- new_cases_per_million: New confirmed cases of COVID-19 per 1,000,000 people. Counts can include probable cases, where reported.
- new_cases_smoothed_per_million: New confirmed cases of COVID-19 (7-day smoothed) per 1,000,000 people. Counts can include probable cases, where reported.
- icu_patients	Number of COVID-19 patients in intensive care units (ICUs) on a given day
- total_tests: Total tests for COVID-19
- new_tests: New tests for COVID-19 (only calculated for consecutive days)
- total_tests_per_thousand: Total tests for COVID-19 per 1,000 people
- new_tests_per_thousand: New tests for COVID-19 per 1,000 people
- new_tests_smoothed: New tests for COVID-19 (7-day smoothed). For countries that don't report testing data on a daily basis, we assume that testing changed equally on a daily basis over any periods in which no data was reported. This produces a complete series of daily figures, which is then averaged over a rolling 7-day window
- new_tests_smoothed_per_thousand: New tests for COVID-19 (7-day smoothed) per 1,000 people
- positive_rate: The share of COVID-19 tests that are positive, given as a rolling 7-day average (this is the inverse of tests_per_case)
- tests_per_case: Tests conducted per new confirmed case of COVID-19, given as a rolling 7-day average (this is the inverse of positive_rate)
- total_vaccinations: Total number of COVID-19 vaccination doses administered
- people_vaccinated: Total number of people who received at least one vaccine dose
- people_fully_vaccinated: Total number of people who received all doses prescribed by the initial vaccination protocol
- total_boosters: Total number of COVID-19 vaccination booster doses administered (doses administered beyond the number prescribed by the vaccination protocol)
- new_vaccinations: New COVID-19 vaccination doses administered (only calculated for consecutive days)
- new_vaccinations_smoothed: New COVID-19 vaccination doses administered (7-day smoothed). For countries that don't report vaccination data on a daily basis, we assume that vaccination changed equally on a daily basis over any periods in which no data was reported. This produces a complete series of daily figures, which is then averaged over a rolling 7-day window
- new_people_vaccinated_smoothed: Daily number of people receiving their first vaccine dose (7-day smoothed)
### Three graphs of EDAs:
<img width="688" alt="image" src="https://github.com/LoveYourself999/End-to-End-MLProject/assets/54390035/e4198e5b-233c-47af-a847-e6e04f2103ee">


The graph illustrates the frequency distribution of our target variable, 'high_risk,' which represents the ICU patient ratio in the hospital. As evidenced by the graph, the most common value for our target variable is approximately 0.20 and 0.28.

<img width="825" alt="image" src="https://github.com/LoveYourself999/End-to-End-MLProject/assets/54390035/66699bd2-9334-472e-91c0-cc4e92e57769">

The pair plot shows the correlation between each feature, suggesting a potential linear relationship between the following pairs: 
- total_test – people_fully_vaccinated
- total_test — total_boosters
- total_test — people_vaccinated
- people_vaccinated — people_fully_vaccinated
- people_vaccinated — total_boosters
- people_vaccinated — total_test
- total_boosters — positive_rate
- total_boosters — people_fully_vaccinated
##### The correlation between target ‘high_risk’ and one of three features (total_boosters, total_cases, people_vaccinated):
<img width="1519" alt="image" src="https://github.com/LoveYourself999/End-to-End-MLProject/assets/54390035/d752dcc8-a714-4538-be7b-a6f7de66f0bd">  

- The first graph illustrates the relationship between the total number of booster vaccinations and high-risk patients. From this graph, it is readily apparent that as an increasing number of people are vaccinated, the number of ICU patients correspondingly decreases, thereby demonstrating the effectiveness of medical treatments.  
- The second graph presents a detailed view of the relationship between the total number of confirmed cases and high-risk patients. It appears that a certain factor or intervention has significantly lowered the proportion of individuals who, after being confirmed with the virus, end up requiring intensive care unit treatment.  
- The last graph displays the relationship between the number of people vaccinated and the high-risk patients. As the number of vaccinations increases, the harm the virus poses to humans gradually diminishes, particularly among those who have received the second dose, where the number of patients admitted to the ICU plummets dramatically.

## 3. Data cleaning and preprocessing.
#### Data cleaning

To improve our training efficiency, we must remove duplicate rows in our dataset. We begin by examining the data frame for repeated data and confirm that there are no duplicate rows.

We then proceed to address missing values in our features. If a feature has a significant number of missing columns, we make the decision to discard it. This step aims to maintain a balanced dataset for effective training. Since we focus on using vaccination data and testing data, we delete other unrelated features.

It is noteworthy that some vaccine-related data entries have missing values. Yet, we retain them as they correspond to a pre-vaccine availability period (a period when the vaccine was not yet available.) We will fill them with the mean through creating a pipeline that will also scale the features and perform encoding in the next step.

#### Preprocessing

Then we preprocess the data by creating a pipeline that includes:
- Fill in the missing numerical values with the mean using a SimpleImputer.
- Scale the numerical columns using StandardScaler.
- Exclude the target as remainder and pass through the pipeline.
##### The pipeline looks like this:

<img width="348" alt="image" src="https://github.com/LoveYourself999/End-to-End-MLProject/assets/54390035/0672a89f-c259-4d25-b21d-a2f3865c66b7">

Then apply the preprocessing pipeline on the dataset.


## 4. Training and evaluation of three machine learning algorithms, analyze findings, and compare results.
Firstly, we split the data into 80% training set and 20% testing set.
The dimension of the matrix is:
- X_train = (435, 21)
- X_test = (435,)
- y_train = (109, 21)
- y_test = (109,)

Note that (Rows, Columns) Row = number of samples, Columns = number of features
435 samples for the training set, 109 samples for the test set.

### 1. Train a Linear Regression Model:

Apply the LinearRegression as the training model, first train a Linear Regression model with no regularization. Next, use the training data to fit the model and the testing data to predict the value. In order to obtain a more reliable estimate of a model's performance and reducing variance, we applied KFold cross-validation with 5 folds, and reported on the cross validation score, using negative mean squared error as the cross validation metric. Finally, compare the model using MSE and plot the Prediction vs. Actual graph.

Training Process:
- Set the model = LinearRegression()
- Use X_train, y_train to fit the model
- Use X_test to predict values
- Measure the MSE using predict value and labels ‘y_test’.
- Apply KFold cross-validation with 5 folds, and report on the cross validation score
plot the Prediction vs. Actual graph.

### 2. Train a SVM model

Apply the SVR as the training model, setting the maximum depth of the tree to six. In order to obtain best performance, we tune the hyperparameter using GridSearchCV and use the training data to find the best parameters, and then use the parameters we found to predict the values. Finally, evaluate the model using MSE and plot the Prediction vs. Actual graph.
The best parameters is:  {'C': 0.1, 'gamma': 'scale', 'kernel': 'poly'}

Training Process:
- Set the model = SVR()
- Use GridSearchCV to find the best parameters
- Use X_train, y_train to fit the model
- Apply best parameter to SVR model
- Use X_test to predict values
- Measure the MSE using predict value and labels ‘y_test’.

### 3. Train a Decision Tree model:

Apply the DecisionTreeRegressor as the training model, setting the maximum depth of the tree to six. Next, use the training data to fit the model and the testing data to predict the value. Finally, evaluate the model using MSE and plot the Prediction vs. Actual graph.

Training Process:
- Set the model = DecisionTreeRegressor(max_depth=6)
- Use X_train, y_train to fit the model
- Use X_test to predict values
- Measure the MSE using predict value and labels ‘y_test’.
Plot Prediction vs. Actual graph.

Evaluate the performance of three machine learning algorithm by looking at their MAPE:

Since this is a regression task, we choose the model with the lowest MAPE as our best algorithms.

- MAPE for Linear Regression: 12.210558590105638
- MAPE for Decision Tree: 4.317642732746094
- MAPE for SVM: 21.007464195799287

We can conclude that Decision Tree has the lowest MAPE, hence, it is the best algorithm.

## 5. Three graphs for the best performing algorithm.
**scatter chart:**


<img width="729" alt="image" src="https://github.com/LoveYourself999/End-to-End-MLProject/assets/54390035/9953e0db-6636-464f-b5f3-e681df2080a5">


The image provided is a scatter plot generated by the code that visually displays a spread of points. Ideally, if the predictions were perfect, all points would align along a 45-degree line where Predicted Values equal Actual Values, and deviations from this line would indicate prediction errors.  As depicted in the image, an increase in predicted values is correlated with an increase in actual values that infers the model has some level of predictive accuracy. 

**Learning curve:**
<img width="953" alt="image" src="https://github.com/LoveYourself999/End-to-End-MLProject/assets/54390035/e368c24e-25d2-44e9-8bbb-a6b27ffa614f">



The training score (red) starts off high (indicating a low error) and remains relatively constant as more training examples are added, which is typical for a model that is not overfitting.
The cross-validation score (green) improves quickly with more data but then levels off, suggesting that adding more data beyond a certain point doesn't significantly improve model performance.
The convergence of the training and cross-validation scores, along with the low error, suggests that the model is generalizing well. However, the slight gap between the curves indicates there might still be some benefit to adding more training data or adjusting the model complexity for better generalization.

**The tree diagram looks likes this:**

<img width="1266" alt="image" src="https://github.com/LoveYourself999/End-to-End-MLProject/assets/54390035/b478ac31-bf6d-4c20-a8e7-32efe351839a">


### 6. Limitations

We have run into the following limitations:
- The first one is data quality and completeness. We know the quality of data is crucial for accurate predictions. But our data comes from the internet which is an open source website. It might have inconsistencies, missing values, or errors which will lower the accuracy of our predictions.
- The second one is changing pandemic dynamics. The mutations of COVID-19 virus can change rapidly (e.g., new variants of a virus), which might make historical data less relevant for future predictions. 
- The third one is resource availability. The actual number of ICUs needed can depend on resources like ventilators, medical staff, and space, which might not be adequately captured in the data. Meanwhile, the data may not represent all regions equally. Different countries or areas might have different healthcare capacities and policies affecting ICU needs.
  
### Link to the dataset (owid-covid-data.csv)：
https://github.com/owid/covid-19-data/tree/master/public/data 
