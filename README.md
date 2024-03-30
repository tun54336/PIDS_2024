# Volleyball CV

## Problem Description
Current Volleyball Analytics software solutions are unavailable for the broad audience because of their cost and access restricted only to teams that can afford existing software. Ball tracking is an important component for designing tools that help analyze players’ performance on the court.

The task we will be performing is called localization in Computer Vision, meaning identifying the location of an object in the video. The data we are using comes from Temple Division I Volleyball game videos and was obtained with the permission of the team's coach. Our team will train the Supervised Machine Learning Model YOLOv8 on manually annotated video frames to achieve high-accuracy ball tracking. To deal with occlusion occuring during the play we plan on utilizing the Sequential Analysis approach. After achieving the successful ball-tracking, we will move to tracking players positioned the closest to the ball (See: Image Below). This is a significant milestone to offer other developers to build off our foundation and allow identification of elements performed by players such as Set, Pass, or Attack, relevant to the development of volleyball analytics software.

## Motivation
Volleyball Video Analytics Software is an increasingly popular tool used by teams in the United States. It allows players and coaches to watch past games and practices to gain insights about potential tactics allowing them to improve their technical maneuvers. Users of this software can take advantage of the features that make the analysis process more efficient and accurate. For example, coaches can apply filters to analyze missed serves throughout the game.

As the commercially-available solution used by most volleyball teams currently costs $10,000 per year, and is available only to the paying teams, not all the volleyball players have access to this tool. We intend to create a Volleyball Video Analytics model that would serve as a free resource for players and teams who may not have the resources to take advantage of the for-pay product.

## Scope
### 1. What data science solutions are we trying to build?
We propose a Semi-Supervised Computer Vision Model that will track the volleyball and players (See: Problem Description) from video clips provided by the user.
If we can succeed with this we will expand our scope to include annotation of additional volleyball elements
* Player Action
* Referees
  
The goals of this solution include: tracking the ball, accurately annotating volleyball elements, and potentially dealing with occlusion happening during the play.

The model will utilize supervised machine learning techniques, including Convolutional Neural Networks, Computer Vision Models (YOLOv8), and Computer Vision Libraries (OpenCV).

### 2. What will we do?
#### Data Collection and Preparation:
We will collect and annotate the videos of volleyball games from Temple Volleyball team, the data we are annotating will be the volleyball itself and the players which will be identified based on identifiable player numbers.

Every team member will annotate at least 600 frames in order to adopt semi-supervised labeling approach with the rest of the data (See Data Preparation in Methodology) then we will prepare the data to fit into the model .
#### Model Development: 
We will apply several different tracking models (YOLOv8, Tracknet, LSTM, etc) to the annotated data to track the objects and compare their performance. 

As we know that one of our predicted risks will be occlusion, we will apply the Kalman filter to accommodate it and increase the performance. (See Methodology)

#### Evaluation and Validation:
We will evaluate the performance of our overall model using common Computer Vision metrics such as Precision, Recall and Intersection Over Union. 

While the first two metrics are standard in Computer Vision, the last one will be particularly useful in our case because it will help us estimate how close the bounding box fits the prediction in comparison to the ground truth.
We will evaluate the performance of the Kalman Filter using Root Mean Square Error(RSME) and Kalman Gain(K). (See: Methodology: Overall Metrics).

### 3. How is it going to be consumed by the customer?
Currently, we are on pace to have our product be consumable through GitHub. However, a desktop app would be most applicable, and were it to come into scope we would like to begin development on that front.


### 4. Limitations
Our model will be trained primarily on the videos of women's college volleyball which might introduce bias
eg. towards players with ponytails or particular camera angles with the court.

## Methodology

### Data Preparation:
We extracted and manually annotated at least 3000 frames from 3 different recordings using OpenCV(See: OpenCV) and YOLOLabel software.
The annotated frames consist of:
* Frames with ball
* Frames without ball
* Frames with completely occluded ball
* Frames with partially occluded ball

### Semi-Supervised Labeling using YOLOv8(See: YOLO):
We will use YOLOv8 to automate the labeling process of additional 10,000 frames to increase the amount of data for model training. 

We will train the model on manually labeled data and test the performance to see if we need to annotate more data to increase the accuracy.
Then we will sift through the resulting frame predicted by the trained YOLOv8 model, add the correctly labeled frames back into the training data, and retrain the model. We plan to repeat this process until we have enough data for model building.

### Target Variable:
#### Ground Truth:
The bounding box that will encompass the ball in each frame.
* If the ball is not present in the frame and it is not occluded, then there is no bounding box.
* If the ball is not present in the frame and it is completely occluded, then there is no bounding box.
* If the ball is present but partially occluded, then there will be a bounding box encompassing the visible portion of the ball. If the ball is occluded there will be a bounding box surrounding the visible portion of the ball.

### Metrics of Evaluation:

IOU(Intersection over Union)(Area of Overlap/Area of Union) will be used to evaluate the correctness of the ball and the player's location detection.

IOU is the comparison between the predicted bounding box and the ground truth bounding box. It is calculated as the ratio of the area between the intersection of the two bounding boxes to the area of their union.

### Implementation:
We will run our model on testing videos to obtain the predicted bounding box. Then, for each frame of the video, we will calculate the IOU with the corresponding ground truth bounding box.

### Threshold:
IOU > 0.5 - True positive(TP) - The model correctly identifies the volleyball in each frame and the prediction box overlaps with most of the ground truth box.

IOU < 0.5 - False Positive(FP) - The model incorrectly identifies the volleyball in each frame, and the prediction box overlaps with less of the ground truth box.

False Negative(FN) - There is no explicit threshold, but it is the case when the model fails to detect the ball when it is present according to the ground truth data

True Negative(TN) - There is no explicit threshold and it is not applicable in object detection models, since the objective is to detect existing objects, rather than to detect non-existent objects.

### Evaluation:
To get an idea for the overall model performance, we will find the average IOU of all the frames. A higher average means better performance, while a lower IOU means there is more need to refine the model.
Sequence analysis
### Kalman Filter:
A popular mathematical prediction algorithm that uses a series of (i.e. 1, 2, 3, …, n) positional data to produce an estimate on the next position (i.e. n+1, n + 2, …)
* https://medium.com/@ab.jannatpour/kalman-filter-with-python-code-98641017a2bd
* https://pieriantraining.com/kalman-filter-opencv-python-example/
* https://www.hindawi.com/journals/wcmc/2022/8280428/

The Kalman Filter will assist in predicting the position of the labeled objects when it is occluded.

### Metric of Evaluation:
#### Root Mean Square Error(RMSE):
We will measure the average magnitude of the difference between the estimated position of the ball predicted by the Kalman filter and the ground truth position of the ball. A lower RMSE value indicates better performance of the Kalman Filter.
* https://kalman-filter.com/root-mean-square-error/

#### Final Model Metrics:
* Precision [TP/(TP+FP)]

Precision is defined as the proportion of correctly identified instances of a volleyball element out of all instances identified as a volleyball element by the model. A high precision indicates that the model has a low false positive rate and it doesn’t mistakenly detect another object
#### RecallTPTP+FN
Recall is defined as the proportion of correctly identified instances of a volleyball element out of all instances of volleyball elements in the data.A high recall indicates the model has low false negative and it is able to detect the volleyball elements when it is present in the data

## Plan

As of 3/20 we have labeled over 2500 frames for initial training of Yolo balanced the dataset to mitigate False Positives. As of now, we are moving forward with using the model to assist us in labeling more frames. (See: Architecture; Yolo)

We currently have preliminary results for our First Deliverable as of 3/19.
* Ball Tracking (See: Architecture; OpenCV)

We intend to work on minimizing the False Positives in our model and have a more refined model reflecting these changes by 3/26.

Our Second Deliverable is intended to be finished by 4/17.
* Human Action Recognition

Currently, to not go beyond scope we intend to track and mark Active Players rather than all players as this would muddy the model with high levels of occlusion. Active Players are noted by interaction with the ball.
Final Deliverable and Demo to be presented on 4/26

### Future Work
With the current progress made on our model, we are interested in if this means of labeling is the most effective for tracking both the ball and players.As a result, if time permits, we would like to potentially work on an alternative form of labeling known as segmentation to see if there is a notable use between the two.

## Architecture
### Data & Data Preparation

#### Primary Source:
We will be using recordings of college volleyball games provided with consent by a Temple Volleyball Coach.There are six videos total with possibility of acquiring more data if necessary
The videos range from 2 hours to 3 hours in length with all but one video ranging above 2 ½ hours.
The data size averages at 4 GB with one outlier around 6 GB.

#### Secondary Source:
Recording of professional volleyball games provided at the YouTube channel 
* https://www.youtube.com/@MMGVolley

These videos are on various courts, but all are from a constant vantage point.
We have decided this will be a secondary source as with consent of use in our Primary Source we can use it in portfolios. Were we to use our Secondary Source, as it is from YouTube the data belongs to the copyright holder and cannot be utilized outside of this class.
#### Data Quality:
Each recording acquired by our teammate was encoded with 30 frames per second. The resolution is consistently 1920 x 1080, with an aspect ratio of 16:9
Of the six videos:
The balls being used are consistent in color.
3 of the videos are on the same court with the remaining being on different courts.
All videos have a constant vantage point.

#### GPU
Google Cloud Host GPU; NVIDIA Tesla T4
* 1 GPU
* 16 GB of GDDR6 memory

#### Google Colab Vs. Cloud Host

GPUDeep learning is a GPU-intensive endeavor, and with the use of the YOLO algorithm, it is even more necessary to seek a GPU cloud host for training on our data.

There is limited use for Google Colab GPU before having to upgrade to a premium version
* Disadvantage: The premium is charged by a “compute unit” which is not measured by time usage or given a proper measurement making estimating a budget in turn harder to determine.
* As a result, we believe Google Colab will be beneficial to run lighter loads to reduce time waste and costs.

Using the NVIDIA T4 model - After researching our options, we agreed on the use of this GPU model as the good balance between high computing capabilities and a reasonable price.
* GCP currently charges the NVIDIA T4 model at $0.35/hr. Based on our calculations our budget should cost between $53 and $88. (See Budget for Details).

#### YOLO
You Only Look Once is a common Machine Learning model used in Object Detection tasks.
##### Labeling Optimization
* We propose a semi-automated approach to labeling with the YOLO model that will allow us to gather more training data in a shorter amount of time.
##### Efficient Object Detection
* YOLO processes video frames through a single pass through a neural network, which enables real-time object detection which is ideal for video streams.
##### High Accuracy
* YOLO divides each of the video frames into grids, and predicts the bounding box and the class of the object in each of those grids, leading to accurate object localization and classification which is crucial in labeling the ball accurately in each frame of the volleyball video.
  
#### OpenCV
Open Source Computer Vision Library is used for image processing, video frame extraction, and video analysis tasks.
Using VideoCapture class to read a variety of video data types (i.e. AVI, mp4)
* Using the read() method, we can extract the frames of the selected video.
* Extract frames from each of the videos at regular time based intervals
* If the video has high frame rate, the regular interval would be longer
* If the video has lower frame rate, the regular intervals would be shorter
##### Error Handling:
Using functions in OpenCV, we can check whether each of the processes, from reading to extraction, is performed correctly.
##### Data Movement/Storage
The video data is stored on Google Drive and accessed by team members.
We will potentially, move it to the Google Cloud Platform after labeling when the amount of labeled video frame gets too large
Since the video data is currently stored in Google Drive, the Google Storage transfer service makes it easy to move our data.

#### Platform
The client will be able to access the project through a GitHub repository.

## Communication
Currently, the general roles are as follows:
* Magda: Team lead, Feature Engineering, Model Building
* Kat: Research, Labeling, Model Building
* Jefferson: Feature Engineering, QA, Model Building
  
There will be two meetings a week:

The first meeting will be early in the week and is a collaborative meeting to address what each team member will need to do before the next meeting.

The end-of-the-week meeting will be a check-in to see how weekly progress went and what targets were hit and missed. We will then try to see if the missed targets can be solved in a group setting before the next meeting.
To enhance communication and assure team member participation and satisfaction we have a text chat in which members can send feedback and updates on the project at any time to address quick fixes and prioritize during the end-of-week meeting.

## Risks
### Budget
Referencing a run of our YOLO model to label the ball without GPU, it took about 11 hours. This model ran on a T4 GPU which took a little over an hour.
With the knowledge our model usage will grow more complex than this we have averaged a usage of 30 to 50 hours a week resulting in our budget range.

### Labeling
Computer Vision Projects generally require large amounts of data for Model Training. We propose a solution to speed up the data annotation process. (See: Yolo; Architecture.)
### Occlusion
This is a common problem in Computer Vision and currently has no foolproof solution. It does however have several proposed approaches to minimize the accuracy issues attached to it.
Below is a paper proposing using the tracked ball trajectory to estimate areas where the ball will appear once the period of occlusion ends. It claims when compared to similar models it surpassed them.
* https://ieeexplore.ieee.org/document/7797015



