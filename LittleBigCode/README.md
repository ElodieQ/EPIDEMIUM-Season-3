# Epidemium 3 - LittleBigCode team

Can the positive stainings corresponding to the cells composing the tumor microenvironment be predictive of patient survival ?

## Repo organization

Poster Epidemium LittleBigCode.pdf

data/<br>
&nbsp;&nbsp;&nbsp;&nbsp;KORL_avatar_train.csv<br>
&nbsp;&nbsp;&nbsp;&nbsp;KORL_avatar_train.csv<br>
&nbsp;&nbsp;&nbsp;&nbsp;marker1/ : contains all .jpg images of marker 1<br>
&nbsp;&nbsp;&nbsp;&nbsp;marker2/<br>
&nbsp;&nbsp;&nbsp;&nbsp;marker3/<br>
&nbsp;&nbsp;&nbsp;&nbsp;marker4/<br>
&nbsp;&nbsp;&nbsp;&nbsp;marker5/<br>
&nbsp;&nbsp;&nbsp;&nbsp;marker6/<br>
code/<br>
&nbsp;&nbsp;&nbsp;&nbsp;ppc.py : Gather the data in a usable way and preprocessing code<br>
&nbsp;&nbsp;&nbsp;&nbsp;train_dl.py : Everything related to the DL approach (DataModules, Models, functions, ...)<br>
notebooks/<br>
&nbsp;&nbsp;&nbsp;&nbsp;hp_study.pkl : One of the Optuna optimisation results we did<br>
&nbsp;&nbsp;&nbsp;&nbsp;bagnet_marker2.pt : A trained BagNet state_dict for the marker 2 (check notebook 05 for how to use it)<br>

## Our Approach
The objective we have set is to predict the class of overall survival (short, medium or long survival) of a patient with ORL cancer due to human papillomavirus (HPV). 
To deal with this multiclass classification problem, two approaches were tested : 
- A machine learning approach considering the quantity of antibodies
- An approach using Deep Learning on the images to get further insights.

N.B: We have also submitted predictions on a Naïve DecisionTree predicting OS from Alcool, Tabacco and OMS score

## Machine Learning Modeling and Interpretability

We don't user the raw images but we calculate the percentage of red per image. We use Random Forest and XgBoost models and we did tests with just the images or with both images and clinical data. <br>

The whole process is presented in the notebook notebooks/03-image_combinations-simple-ppc.ipynb

## Deep Learning Modeling and Interpretability

For Deep Learning approach, two methods :  

- Using multiple markers as input : Taking only the red channel of each marker image and directly feeding these to a deep learning model that will learn its own features. For this method, we used a custom basic CNN, due to time constraints. We mainly focused on the second approach.

- Using one single marker as input : Taking the full RGB image as input. For this method, we used for each marker indivdually a BagNet model in order to have a solid interpretability part. 

The code is in code/train_dl.py.<br>
Everything is presented synthetically is the notebook notebooks/05-deep-learning-modeling.ipynb

## Results 

To analyze the results of our multiclasses classification models, we split our training dataset in two parts : 70% dedicated to model training and 30% to model validation (this represents 12 patients).
Results are not very stable given the small amount of data.

Our best model could predict the correct category on 8 out of the 12 validation patients : 66.7% accuracy.

## References

https://github.com/wielandbrendel/bag-of-local-features-models : BagNet repo<br>
https://openreview.net/pdf?id=SkfMWhAqYQ : BagNet paper
