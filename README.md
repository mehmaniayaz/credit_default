# Table of Contents
1. [Summary](#summary)
2. [Dataset](#dataset)

## Summary <a name="summary"></a>
We are interested in the determining the likelihood of a client to default on their credit card. This is therefore a classification
problem with features including numerical, categorical, as well as ordinal. The repo contains a notebook that you can use to 
select features and tune hyperparameters conveniently. 

<p align="center">
<img src="./results/testing/confusion-matrix-test.png" width="80%"/>
</p>
Figure 1: Test set confusion matrix. Results are normalized by the true values for each class.

<br></br>


<img src="./results/testing/feature_importance.png" width="40%" align="center"/><img src="./results/testing/learning_curve.png" width="40%" align="center"/>

Figure 2: Feature importance and learning curve for the trained model. The latter illustrates our model and dataset are prone to overfitting.


## Dataset <a name="dataset"></a>
You can find the data [here](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).
