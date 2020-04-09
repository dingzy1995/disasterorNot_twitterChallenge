# disasterorNot_twitterChallenge
This is a repository for the final project of COSC 572 spring 2020

COSC572 Final Project Proposal			Shuyang Yu & Ziyao Ding
Our project is based on a Kaggle challenge Real or Not? NLP with Disaster Tweets https://www.kaggle.com/c/nlp-getting-started/overview
Topic
Identify whether tweets are announcing real disasters or not, with generative model and further experiments based on them.
Background info
Twitter has become an important communication channel in times of emergency.
The iniquitousness of smartphones enables people to announce emergent situations they’re observing in real-time. Because of this, more agencies are interested in programmatically monitoring Twitter (i.e. disaster relief organizations and news agencies). But, it’s not always clear whether a person’s words are actually announcing a disaster.
Dataset and methods
Training and test sets are provided by Kaggle, and each contains 10,000 tweets.
	First, a method would be built to complete the classification problem described on Kaggle. According to the information we collected up to now, it will probably be a combination of a tf-idf index combined with a pre-trained deep neural network model.
	Then, we would try to dig into the possible grammar structure and other possible linguistic factors that make a tweet announcement of disaster. Hopefully, we could work out a scoring model that tests how likely it is a tweet is a disaster using the training data provided.
	Next, we plan to implement a generative model that generates some ‘fake’ tweets that contain disaster keywords. We would use them as the input to the model mentioned in step 2 to get a score for each of them.
	Lastly, we would try to combine the generated dataset with the original one and use them in the classification model in step one. Observations and analysis would be produced.
Evaluation
As for the classification task, Kaggle has evaluate system itself.
As for the generative task, for now we decide to use our common sense to determine the performance, and the analysis later will explain more.
Division of work and timeline
We would complete the preprocessing, cleaning, and feature engineering in approximately one week.
Then, we would divide our work by the two models and complete both of them in about two weeks.
The last week would be used to implement the analysis and other further thoughts or situations.
