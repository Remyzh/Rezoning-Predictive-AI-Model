# Rezoning-Predictive-AI-Model
A machine learning model that predicts the outcome of rezoning applications as "approved" or "rejected" based on various application attributes.

-By Remy Zhang

### Introduction

In Canada's major cities like Vancouver, the ongoing housing crisis is characterized by a significant demand for housing and limited supply. To increase the housing stock, constructing high-density housing has been identified as a potential solution that can accommodate more people on the limited amount of land available in urban areas. However, the creation of high-density housing requires the rezoning of low-density, industrial or farmland, a difficult process that can take months or even years to complete. This process involves multiple public hearings, environmental impact reports, expertise in planning law, research of the city council, and financial reports, making it a costly and time-consuming endeavor for developers. From a report published by the National Multifamily Housing Council, when rezoning is required to develop multifamily high-density housing, rezoning efforts represent 3.4% of the entire cost of development (NMHC, 2019). Considering that high-density developments like multifamily or high–rises can cost tens of millions to develop, this is a large percentage cost. Even with all these efforts, there is still no guarantee that the application will be approved, posing a big risk factor for developers in building more housing. 

Due to the cost and time required to complete the rezoning process, the lack of available land, and community resistance to new developments, developers are often forced to focus on constructing high-rise buildings, leading to a "missing middle". Where there is a lack of low-rise high-density housing, which offers better aesthetics and is quicker and cheaper to construct per unit.

The construction of a high-rise building can take anywhere from 2 to 5 years, while the construction of a low-rise building of 5-6 stories can take anywhere from 6 months to 2 years (CAT, n.d.). However, due to the high cost of rezoning applications, developers tend to prioritize high-density construction, limiting the amount of new low-rise housing supply.

To address these housing issues, a machine learning model that predicts the outcome of a rezoning application as approved or rejected could potentially enable developers to enter specific information about rezoned land and determine whether it is suitable for rezoning without the need for massive research efforts. Such a model could reduce the time and capital costs of rezoning applications, helping to increase the supply of new low-rise and high-rise housing. 
The dataset used in this project contains all 162 instances of tall residential buildings constructed in Toronto from 2008 through 2018 recorded by the Council on Tall Buildings and Urban Habitat. The target variable would be the rezoning result, which is not available but can be derived from available data and this will be discussed later in this report. Below are the inputs from the dataset:

● Status (under construction, complete, proposed) - string

● Name of Tower - string

● Use (residential, office, retail) - string

● Address - string

● City - string

● # floors - discrete int

● Proposed Date- string

● Rezoning Authority (OMB, City Council) - string

● OMB Case # - string

● Type of Approval (Null, Hearing, Settlement, prehearing) - string

● Rezoning approval date - string

● Rezoning duration - int


### Background 

The core problem that my project aims to solve is to classify the features of a rezoning application to make a binary prediction of whether or not that application will be approved or rejected. There is currently a paper titled: “Evaluation of Current Construction Permitting Process in City of Toronto and Future of Permitting in the Global
Construction Industry” (Shahi, 2018), implementing a Bayesian Model, which, similar to my project, predicts the outcome of rezoning applications. In this paper, the Bayesian model was able to achieve an 84.6% accuracy. However, no snippets of code or documentation of hyperparameters were recorded on the paper. This paper was the only one which researched the application of machine learning in rezoning prediction, so, I looked at literature concerning tabular datasets and binary classification as a whole.

Current literature suggests that Gradient Boosted Tree (GBT) models like XGBoost are considered one of the best machine learning models when working with Tabular datasets, citing their advantage of high accuracy and robustness for removing outliers and noise while focusing on feature importance (Oleksandr, 2022). However, gradient-boosted tree models like XGBoost require larger datasets than what my dataset has, the recommended number of observations should be greater than 500 rows, but my dataset only contains 151 rows. Due to the lack of data to produce an accurate and non-overfitting GBT model, I decided to employ ML models that traditionally work better with smaller datasets, comparing and contrasting the results of a Logistic Regression model, Random Forest, SVM Model, then conducting hyperparameter tuning on the model that performs best.

Notably, neural networks also perform well with tabular datasets, and in many cases are documented to perform better than Gradient Boosted Trees (DataRobot, 2021). Thus, in this project, I also tested the data on a Multilayer perceptron (MLP) model and conducted hyperparameter tuning while graphing test vs train accuracies to find an optimal set of parameters that don’t overfit.



### Data Collection and Cleaning

Starting with Data collection, the dataset used in this project contains 162 instances of tall residential buildings constructed in Toronto from 2008 through 2018 recorded by the Council on Tall Buildings and Urban Habitat. An article discussing the relationship between rezoning and school locations was established and this inspired the creation of a new column called “Distance to School” which outlined the distance in meters from the rezoning address to the nearest school (Calder, 2019). This was done manually on Google Maps, searching each address and then measuring the distance to the nearest school. The distance to major roads is also a factor that could affect rezoning success. Typically major roads carry more infrastructure like water lines and sewage which makes construction less costly for a city, compared to if the rezoning development was rural. This then inspired the creation of another column named “Major road?” which took the street name of each rezoning application and cross-referenced it against Toronto’s list of Major Arterial roads (Geographic.org, n.d.). 

To better understand the importance of the different features and better eliminate the non-relevant features, we conducted some data analytics. As a quick and intuitive way of identifying patterns between the features and the target variable, a heatmap was developed. Notably, since a heatmap is not good at identifying non-linear relationships, SHAP (SHapley Additive exPlanations) was used to further explore the feature relationships.


An issue with the dataset from the CTBUH is that it does not include labels for the application status of each rezoning application specifying them as “rejected” or “approved”, leaving us without the crucial target variable. After research, it was discovered that every occurrence of an OMB case number means the application has been rejected at least once.  So, if the status of a rezoning application is proposed and there exists an OMB case number, we can label these applications as “rejected” inside a new column called “rezoned status”. Moreover, if an application has the status “built” or “under construction”, these applications would be labelled as “approved”. At this stage, rezoning applications with more than three pieces of crucial information like “rezoning status” and “rezoning date” were removed. Other non-relevant columns were also removed.

Some data like the “Use” data feature has five unique options telling us what the intended use of the land is. One-hot-encoding was conducted and encoded the various options into “Residential”, “hotel”, “office” “education” and “retail”. Similarly, Binary Encoding was conducted to encode the “Rezoning Authority” feature into a binary 0 representing “OMB” or 1 representing “City Council”.








### Model Design:


In this project, I conduct hyperparameter tuning on the two models: Random Forest and Multi-layer Perceptron. Starting with Random Forest, I test the following parameters and then return the combination with the highest test accuracy.

Hyperparameter
Values to test
n_estimators
[10, 30, 50, 100]
max_depth
[10, 20, 30]
min_samples_split
[2, 5, 10]
min_samples_leaf
[1, 2, 4]


Depending on if the accuracy values lie on the edges of the starting hyperparameters (Ie. n_estimators = 10 or 100), I may choose higher or lower values for  further tuning. Then, when the set of hyperparameters with the highest test accuracy is found, the test and train accuracies will be compared to check for overfitting, making sure train accuracy does not surpass test accuracy by a large margin.


Moving onto the MLP model, it consists of 11 nodes in the input layer imputed into a single hidden layer with ReLU activation and then with one binary output node. The reason for designing the MLP with one hidden layer is to avoid overfitting, as there are only 151 rows of data.

Hyperparameter
Values to test
hidden_size
[5, 10, 60]
num_epochs
[10, 20, 30]
learning_rate
[0.001, 0.001 0.0001]
batch_size
[64, 128]



Shown above is the diagram of the MLP model design used in this report. Please note that the size of the hidden layer in the diagram represents just one combination of the hyperparameters, and this diagram may change as hyperparameters change.
Metrics

In evaluating the performance of our trained models with tuned hyperparameters, the primary metric was accuracy. Accuracy was expressed as a percentage indicating the likelihood in test or training data that each trained model was going to correctly predict a rezoning application as “approved” or “rejected”. 


### Results 
Starting with the results of all the individual simple regression and tree models, below are the initial results without hyperparameter tuning:


Model
Test Accuracy
Logistic Regression Model Accuracy
80.65%
Decision Tree Model Accuracy
83.87%
Random Forest Accuracy
87.10% (86.39% train accuracy)


Out of the simple regression and tree models, Random Forest achieved the highest test accuracy. As outlined in the methodology, the next step was to conduct hyperparameter tuning on the Random Forest model, where it was found that the combination of  'max_depth' =10, 'min_samples_leaf'= 4, 'min_samples_split'=10 and 'n_estimators'= 50 produce the highest test accuracy. From hyperparameter tuning, the test accuracy for Random Forest increased from 87.10% to 93.55%. To check overfitting with this tuned Random Forest model, the test and train accuracies were graphed with respect to the number of estimators. As seen, in the graph below, the train and test accuracies are within 1% in accuracy and thus does not show overfitting.



Next moving onto the MLP model, tuning with the starting hyperparameters grid resulted in a test accuracy of 70.96% with hidden_layers=10, num_epochs=150, learning_rate=0.001 performing the best. With respect to hyperparameter tuning, it can be observed that as the number of epochs surpasses the range of 150-200, there is a plateau. Moreover, graphing the testing vs training accuracies, we find the following figure, indicating little to no overfitting observing that the test accuracy is mostly higher than the training accuracy.

Note the above 2 figures are examples of tuning hyperparameters for the MLP model. Comparatively the testing accuracy of the MLP model is lower than the Random Forest model and this result is expected since MLP models are not as effective with a small number of observations and features in a dataset.

### Limitations

As alluded to in the background section, there are a few key limitations to the dataset of rezoning applications which affected the accuracy and thoroughness of this project. Primarily, the dataset from the CTBUH does not include labels for each rezoning application specifying them as “rejected” or “approved”, requiring the target variable definition outlined in the Methodology section. Without target variable labels, my approach can be problematic in the case where a rezoning application was rejected at the time of the dataset, but was able to submit additional appeals to ultimately get the rezoning approved. For example, the development named “Grid Condos” on 181 Dundas Street E, at the time of the dataset collection in 2018 was in the rezoning proposal state with an OMB Hearing number, indicating the application was rejected. However, “Grid Condos” was able to successfully appeal their rezoning application in subsequent years, making the model incorrectly labeled in my data. While in my data processing stage, I manually correct these mistakes, a comprehensive solution would be to secure an up-to-date dataset and to collect a dataset with a pre-labelled column indicating whether each rezoning application was rejected or approved.

Another limitation was the small size of the dataset I had at only 151 rows of data with only 18 recorded features. This combination of a limited number of observations and features made the models overly dependent on a few key features. While the Decision tree model was able to score a high test accuracy with low overfitting, because the dataset was so small, if new data is introduced, the accuracy could drop. Moreover, having only 151 rezoning applications to input, certain higher-performing models could not be utilized to their potential, like MLP and GBT. In the future, I would like to try MLP in addition to a gradient-boosted tree model like XGBoost to achieve a more robust and higher accuracy prediction of rezoning applications.

Additionally, the dataset from the CTBUH only includes “tall buildings” which excludes buildings under 14-stories like low-rise, multi-family and other types of high-density housing in need of rezoning. Therefore, a limitation of my model is that it can only predict the outcome of rezoning applications for tall buildings, and the accuracy would most likely not be carried over to rezoning applications for other forms of high-density housing.


### References
National Multifamily Housing Council (NMHC) & National Association of Home Builders (NAHB). (2019). Cost of Regulations Report. Retrieved from https://www.nmhc.org/research-insight/research-report/nmhc-nahb-cost-of-regulations-report/ 

Canada Mortgage and Housing Corporation (CMHC). (n.d.). Guide to Developing Affordable Housing. Retrieved April 25, 2023, from https://www.cmhc-schl.gc.ca/en/professionals/housing-markets-data-and-research/housing-research/research-reports/nhs-outcomes/guide-developing-affordable-housing 

Shahi, K. (2018). Understanding Housing Affordability in Toronto: A Critical Examination of the City's Policy Response [Master's thesis, University of Toronto]. TSpace. Retrieved from https://tspace.library.utoronto.ca/bitstream/1807/89527/3/Shahi_Kamellia_201806_MAS_thesis.pdf 

Oleksandr, S. (2022, April 20). Tabular Data Binary Classification: Tips and Tricks from 5 Kaggle Competitions. Neptune. https://neptune.ai/blog/tabular-data-binary-classification-tips-and-tricks-from-5-kaggle-competitions

DataRobot. (2021, October 20). AI Simplified: How to Build and Deploy ML Models [Video file]. Retrieved from https://www.youtube.com/watch?v=WPQOkoXhdBQ&t=455s&ab_channel=DataRobot 

Calder, V. B. (2019, November). Zoned Out: How School and Residential Zoning Limit Educational Opportunity. United States Congress. Retrieved from https://www.jec.senate.gov/public/index.cfm/republicans/2019/11/zoned-out-how-school-and-residential-zoning-limit-educational-opportunity 

Geographic.org. (n.d.). Street View of City of Toronto, Ontario, Canada. Retrieved April 25, 2023, from https://geographic.org/streetview/canada/on/city_of_toronto.html 
