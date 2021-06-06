# YouTube-Video-Popularity-Predictor

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Introduction](#introduction)
* [Data Source](#data-source)
* [Tools](#tools)
* [Data Preprocessing](#data-preprocessing) 
  * Feature selection
  * Outlier detection
* [Prediction Models](#prediction-models)
  * Model 1
  * Model 2
* [Results](#Results)
* [Conclusion](#Conclusion)
* [Recommendations and Future Work](#recommendations-and-future-work)

<!-- INTRODUCTION -->
## Introduction

Youtube is the largest video-sharing and streaming website on the internet where users upload, view, rate, share, add to favourites, report, and comment on videos. There is a large possibility for examining data present on YouTube and getting useful insights out of it. This project aims to make an effort to help a huge set of people to predict the popularity of their next video. Influencers can modify the content accordingly and generate greater revenue from the views. There are many features on which the views on the video depends such as subscriber count, title, thumbnail etc. We want to identify those important features and predict the view count on a video using Machine Learning algorithms.

<!--DATA SOURCE-->
## Data Source

Dataset was downloaded from the [YOUTUBE - 8M](https://research.google.com/youtube8m/index.html) website.

<!--TOOLS-->
## Tools
* Python programming language

<!--DATA PREPROCESSING-->
## Data Preprocessing
* **Feature selection:**
* The dataset of information about the videos was saved locally into a CSV file with features: <br/>
*Title, Description, CategoryId, PublishedAt, Current-
Time, Life Definition, Caption, Duration, Dimension,
Latitude, Longitude, LikeCount, DislikeCount, View-
Count, FavoriteCount, CommentCount, Tags, Thumbnail,
ChannelId, ChannelTitle, ChannelSubscribers, ChannelUploads,
ChannelViews, ChannelComments, Country.* <br/>
* Considering features with numerical value only, the following
attributes were decided: </br>
*Duration, LikeCount PreviousVideo, Dislike-
Count PreviousVideo, CommentCount PreviousVideo,
NumberOfTags, ChannelSubscribers, ChannelUploads,
ChannelViews, VideoLength, DescriptionLength, SocialLinks*
* The thumbnail is an important
feature for the predictions, Python's
’PIL’ library is used to convert the image to
grey-scale and the pixel values are stored in numpy arrays.
These images are passed in the convolutional
neural network. The results from thumbnails are used as a
feature to train further models and hence can be added in
the above list.
Features after thumbnail extraction: <br/>
*Duration, LikeCount PreviousVideo, Dislike-
Count PreviousVideo, CommentCount PreviousVideo,
NumberOfTags, ChannelSubscribers, ChannelUploads,
ChannelViews, VideoLength*
* The analysis was made with respect to the importance of
the feature (calculated using the Impurity techniques used
in the Random Forest Classifiers). The following graph has
been observed:
![image](https://user-images.githubusercontent.com/81852314/120921273-93d33b00-c6e0-11eb-8603-249e1b8c5d37.png)
It can be analyzed from the above graph that features
such as Dislike Count, Like Count play a major role in
predicting the results. The other features also influence the
results to a nice extent.

* **Outlier detection:**
![image](https://user-images.githubusercontent.com/81852314/120921307-b8c7ae00-c6e0-11eb-8e4f-77ef668f226e.png)
* The outliers were removed using the Z-Score Algorithm, keeping a threshold
of 4 for the features and 10 for the View Count. 

<!-- PREDICTION MODELS-->
## Prediction Models
* **Model 1:** Initially, the data have been trained on 4 models:
  * Linear Regression
  * Decision Tree Regression
  * Random Forest Regression
  * Support Vector Machine
  The results (accuracies) from these models were
not up to the mark and thus thumbnail is introduced in II
part to improve the output.
* **Model 2:**  The training of the model has been done in 2 parts: <br/>
  * The first part focuses on extracting the feature, ”thumbnail
result”. Convolutional Neural Network has been
used for the same. Dataset for the CNN consists of the
Thumbnails and the labelled views. The actual views
have been mapped into labelled views by classifying the
views into 10 classes. The data is then trained with the
CNN and the resulting labels, giving an approximation of
the actual views are stored as “thumbnail result” to be used
further. <br/>
  **Architecture of CNN:** <br/>
Number of Layers: 6
*Layer1:* Convolution Layer with, kernel size = 5, padding
= 2, stride = 1, dropout = 0.25, mapping one input channel
into 16 channels. <br/>
After this, ReLu is applied on the outputs.<br/>
*Layer2:* Max Pooling Layer with stride = 2 <br/>
*Layer3:* Convolution Layer with, kernel size = 5,
padding = 2, stride = 1, dropout = 0.25, mapping 16 input
channel into 32 channels.<br/>
After this, ReLu is applied on the outputs.<br/>
*Layer4:* Max Pooling Layer with stride = 2<br/>
*Layer5:* Fully Connected Layer mapping into 100
neurons, dropout = 0.25.<br/>
*Layer6:* Fully Connected Layer mapping 100 neurons
into 10 class probabilities. <br/>
  * The ”thumbnail result” feature is now used with
the above-mentioned features and has been trained with the
following models, with the target variable being the actual
number of views.
    * Support Vector Regressor
    * Random Forest Regressor
 ![image](https://user-images.githubusercontent.com/81852314/120922056-aea7ae80-c6e4-11eb-8516-a98ac286be7b.png)

<!--RESULTS-->
## Results
* Model 1:
  * Linear Regression: <br/>
R2 Score on the Training Data: 0.6364104183115034 <br/>
R2 Score on the Test Data: 0.6282098478473409 <br/>
  * Decision Tree Regressor: <br/>
R2 Score on the Training Data: 0.999999740665929 <br/>
R2 Score on the Test Data: 0.5009736557269098  <br/>
  * SVR: <br/>
R2 Score on the Training Data: 0.7536050917458403 <br/>
R2 Score on the Test Data: 0.7255821853378172 <br/>
* Model 2:  <br/>
  * SVR:  <br/>
R2 Score on the Training Data: 0.955  <br/>
R2 Score on the Test Data: 0.92  <br/>
  * RandomForest:  <br/>
R2 Score on the Training Data: 0.97  <br/>
R2 Score on the Test Data: 0.95  <br/>

<!-- CONCLUSION-->
## Conclusion
Thumbnail proved out an extremely important feature
for sure as the results improved significantly.
Seeing the results without thumbnail, only 2 models
were considered most optimal, the SVR and the Random
Forest Regressor.
CNN + Random Forest turns out to be the best model
for this dataset.

<!--RECOMMENDATION AND FUTURE WORK-->
## Recommendations and Future Work
* A simple application could be made using thde developed algorithms
where a user can enter all the features of their upcoming
video having details of thumbnail along with other
attributes.
* Few more important features like
Title of the video and Description of the video can be used as inputs and an NLP Model can be used to predict more reliable results.

