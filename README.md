## Tutorial: Machine Learning with scikit-learn
Presented by [Izaskun Mendia](http://www.dataschool.io/about/) at TECNALIA on Nov 29-30, Dec 1-13, 2016. 

**Hours Allocation Code: Hours Allocation Code: 058585_20200**

### Description

Although numeric data is easy to work with in Python, most knowledge created by humans is actually raw, unstructured text. By learning how to transform text into data that is usable by machine learning models, you drastically increase the amount of data that your models can learn from. In this tutorial, we'll build and evaluate predictive models from real-world text using scikit-learn.

### Objectives

By the end of this tutorial, attendees will be able to confidently build a predictive model, including feature extraction, model building and model evaluation.

### Required Software

Attendees will need [scikit-learn](http://scikit-learn.org/stable/install.html) and [pandas](http://pandas.pydata.org/pandas-docs/stable/install.html) (and their dependencies) already installed. Installing the [Anaconda distribution of Python](https://www.continuum.io/downloads) is an easy way to accomplish this. Both Python 2 and 3 are welcome.

I will be leading the tutorial using the IPython/Jupyter notebook, and have added a pre-written notebook to this repository. I have also created a Python script that is identical to the notebook, which you can use in the Python environment of your choice.

### Tutorial Files

* IPython/Jupyter notebooks: [Titanic_initial.ipynb](Titanic_initial.ipynb), [Tutorial_ML.ipynb](Tutorial_ML.ipynb), [00_pandas.ipynb](00_pandas.ipynb), [01_cleaning_data.ipynb](01_cleaning_data.ipynb),[02_feature_encoding.ipynb](02_feature_encoding.ipynb),[03_about_standardization_normalization.ipynb](03_about_standardization_normalization.ipynb),[04_svm_iris_pipeline_and_gridsearch.ipynb](04_svm_iris_pipeline_and_gridsearch.ipynb),[05_matplotlib_viz_gallery.ipynb](05_matplotlib_viz_gallery.ipynb),[06_linearPolynomicRegression.ipynb](06_linearPolynomicRegression.ipynb),[07_clustering.ipynb](07_clustering.ipynb)
* Datasets: [data/titanic.txt](data/titanic.txt)

### Prerequisite Knowledge

Attendees to this tutorial should be comfortable working in Python, should understand the basic principles of machine learning, and should have at least basic experience with both pandas and scikit-learn. However, no knowledge of advanced mathematics is required.

- If you need a refresher on pandas, I recommend reviewing the notebook of this 3-part [tutorial](http://www.gregreda.com/2013/10/26/intro-to-pandas-data-structures/), and also [00_pandas.ipynb](00_pandas.ipynb).

### Recommended Resources

**Text classification:**
* Read Paul Graham's classic post, [A Plan for Spam](http://www.paulgraham.com/spam.html), for an overview of a basic text classification system using a Bayesian approach. (He also wrote a [follow-up post](http://www.paulgraham.com/better.html) about how he improved his spam filter.)
* Coursera's Natural Language Processing (NLP) course has [video lectures](https://class.coursera.org/nlp/lecture) on text classification, tokenization, Naive Bayes, and many other fundamental NLP topics. (Here are the [slides](http://web.stanford.edu/~jurafsky/NLPCourseraSlides.html) used in all of the videos.)
* [Automatically Categorizing Yelp Businesses](http://engineeringblog.yelp.com/2015/09/automatically-categorizing-yelp-businesses.html) discusses how Yelp uses NLP and scikit-learn to solve the problem of uncategorized businesses.
* [How to Read the Mind of a Supreme Court Justice](http://fivethirtyeight.com/features/how-to-read-the-mind-of-a-supreme-court-justice/) discusses CourtCast, a machine learning model that predicts the outcome of Supreme Court cases using text-based features only. (The CourtCast creator wrote a post explaining [how it works](https://sciencecowboy.wordpress.com/2015/03/05/predicting-the-supreme-court-from-oral-arguments/), and the [Python code](https://github.com/nasrallah/CourtCast) is available on GitHub.)
* [Identifying Humorous Cartoon Captions](http://www.cs.huji.ac.il/~dshahaf/pHumor.pdf) is a readable paper about identifying funny captions submitted to the New Yorker Caption Contest.
* In this [PyData video](https://www.youtube.com/watch?v=y3ZTKFZ-1QQ) (50 minutes), Facebook explains how they use scikit-learn for sentiment classification by training a Naive Bayes model on emoji-labeled data.

**Naive Bayes and logistic regression:**
* Read this brief Quora post on [airport security](http://www.quora.com/In-laymans-terms-how-does-Naive-Bayes-work/answer/Konstantin-Tt) for an intuitive explanation of how Naive Bayes classification works.
* For a longer introduction to Naive Bayes, read Sebastian Raschka's article on [Naive Bayes and Text Classification](http://sebastianraschka.com/Articles/2014_naive_bayes_1.html). As well, Wikipedia has two excellent articles ([Naive Bayes classifier](http://en.wikipedia.org/wiki/Naive_Bayes_classifier) and [Naive Bayes spam filtering](http://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering)), and Cross Validated has a good [Q&A](http://stats.stackexchange.com/questions/21822/understanding-naive-bayes).
* My [guide to an in-depth understanding of logistic regression](http://www.dataschool.io/guide-to-logistic-regression/) includes a lesson notebook and a curated list of resources for going deeper into this topic.
* [Comparison of Machine Learning Models](https://github.com/justmarkham/DAT8/blob/master/other/model_comparison.md) lists the advantages and disadvantages of Naive Bayes, logistic regression, and other classification and regression models.

**scikit-learn:**
* The scikit-learn user guide includes an excellent section on [text feature extraction](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) that includes many details not covered in today's tutorial.
* The user guide also describes the [performance trade-offs](http://scikit-learn.org/stable/modules/computational_performance.html#influence-of-the-input-data-representation) involved when choosing between sparse and dense input data representations.
* To learn more about evaluating classification models, watch video #9 from my [scikit-learn video series](https://github.com/justmarkham/scikit-learn-videos) (or just read the associated [notebook](https://github.com/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb)).

**pandas:**
* Here are my [top 8 resources for learning data analysis with pandas](http://www.dataschool.io/best-python-pandas-resources/).
* As well, I have a new [pandas Q&A video series](http://www.dataschool.io/easier-data-analysis-with-pandas/) targeted at beginners that includes two new videos every week.