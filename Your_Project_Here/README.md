# epidemium3

Can the positive stainings corresponding to the cells composing the tumor microenvironment be predictive of patient survival ?

## Repo organization

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

## Machine Learning Modeling and Interpretability

We don't user the raw images but we calculate the percentage of red per image. <br>
The whole process is presented in the notebook notebooks/03-image_combinations-simple-ppc.ipynb

## Deep Learning Modeling and Interpretability

The code is in code/train_dl.py.<br>
Everything is presented synthetically is the notebook notebooks/05-deep-learning-modeling.ipynb

## References

https://github.com/wielandbrendel/bag-of-local-features-models : BagNet repo<br>
https://openreview.net/pdf?id=SkfMWhAqYQ : BagNet paper
