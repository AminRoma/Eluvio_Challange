# Eluvio_Challange

## Brief Explanation of Methods

### Import packages
```python
import os
import sys
import glob
import json
import pickle
import pickle
import pandas as pd
import base64
import csv
import numpy as np
from google.colab import drive
from glob import glob
from sklearn.metrics import average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
```

### Data preparation and preprocessing

All the .pkl files are put in my google drive. We use a for loop to load .pkl files and load them one by one to data frames. The data frame is updated for each movie and merges all the features stored in tensors to be processed one by one for predictions.  

```python
drive.mount('/content/drive')
data_folder = os.path.join('/content/drive/MyDrive/Colab Notebooks/EluvioData/Eluvio')
filenames = glob(os.path.join(data_folder, '*.pkl'))
gt_dict = dict()
pr_dict = dict()
shot_to_end_frame_dict = dict()
scores = dict()
for fn in filenames:
    x = pickle.load(open(fn, "rb"))


    gt_dict[x["imdb_id"]] = x["scene_transition_boundary_ground_truth"]
    pr_dict[x["imdb_id"]] = x["scene_transition_boundary_prediction"]
    shot_to_end_frame_dict[x["imdb_id"]] = x["shot_end_frame"]

    p1 = pd.DataFrame(x['scene_transition_boundary_ground_truth']).astype("float")  
    p2 = pd.DataFrame(x['scene_transition_boundary_prediction']).astype("float")  
    p3 = pd.DataFrame(x['place']).astype("float")  
    p4 = pd.DataFrame(x['cast']).astype("float")  
    p5 = pd.DataFrame(x['action']).astype("float")  
    p6 = pd.DataFrame(x['audio']).astype("float")


    p = pd.concat([p3, p4,p5, p6, p1], axis=1, ignore_index=True)
    p = p.head(-1)

    X = p.iloc[ : , :-1].values
    y = p.iloc [: , -1].values
```

All of the following steps are done within the for loop for each movie.

### Standardization of features
```python
 imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
 imputer.fit(X[:, :])
 X = imputer.transform(X[:, :])
 sc = StandardScaler()
 X[:,:] = sc.fit_transform(X[:,:])
 ```
    
 ### Reduce the dimensionality of features  
 
I used Kernel PCA to reduce the dimensionality of the features. The preliminary computational result shows that $n=20$ is sufficient to be coupled with SVR or Random Forrest methods. 

```python
kpca = KernelPCA(n_components=20, kernel = 'rbf')
X = kpca.fit_transform(X)```
```
 ### Machine Learning approach
 
I used several methods for prediction and found the best result in using the random forest regression method. The random forest produces the highest scores, and the total computational time for 64 movies is less than two hours. Other methods codes are given, but we only present the scores of the random forest here.

 #### Random Forest Regression
With 20 trees in the forest, we can get AP and Miou scores of >0.99 and >0.95, respectively, for all movies. Thus, increasing the number of trees does not worth it as the computational time increases significantly while the increase in scores is tiny (<0.05).  

```python
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
```
 ##### Support Vector Regression
```python
regressor = SVR(kernel ='rbf')
```
##### Decision Tree Regression
```python
regressor = DecisionTreeRegressor(random_state=0)
```

#### Fitting
```python
regressor.fit(X, y)
y_pred = regressor.predict(X)
y_pred = pd.DataFrame(y_pred)
```
### Save and printing results
```python
pr_dict[x["imdb_id"]] = y_pred
y_pred.to_csv(fn+".csv", sep=',', index=False)
scores["AP"], scores["mAP"], _ = calc_ap(gt_dict, pr_dict)
scores["Miou"], _ = calc_miou(gt_dict, pr_dict, shot_to_end_frame_dict)
print("Scores:", json.dumps(scores, indent=4))
```

### Summary of predictions

In this challenge, I predict the scene segmentation for each movie given features for each shot. The result of scene transition predictions for each movie is saved in the zip file inside the "Eluvio_Challange" directory. There is a CSV file associated with each movie. Inside each CSV file, there is a column that has a value between 0 and 1. The values imply the probability of being the transition boundary for each row of the data frame denoting the movie's shots. 

### Summary of scores for each .pkl file

```python
Scores: {
    "AP": 0.998602448947112,
    "mAP": 0.998602448947112,
    "Miou": 0.9795657170365565
}
Scores: {
    "AP": 0.9990572330799564,
    "mAP": 0.999301224473556,
    "Miou": 0.9831233736918855
}
Scores: {
    "AP": 0.9981523649230603,
    "mAP": 0.9984087935236813,
    "Miou": 0.9782818679990207
}
Scores: {
    "AP": 0.9973650733998687,
    "mAP": 0.9961509174870833,
    "Miou": 0.9718742820174903
}
Scores: {
    "AP": 0.9964508384174533,
    "mAP": 0.9959321492606785,
    "Miou": 0.9686898322178331
}
Scores: {
    "AP": 0.9961331844600347,
    "mAP": 0.9953559437818921,
    "Miou": 0.9655229918008296
}
Scores: {
    "AP": 0.9960314127506149,
    "mAP": 0.9955400028066761,
    "Miou": 0.9625149115464959
}
Scores: {
    "AP": 0.9956370121161046,
    "mAP": 0.9953628689798097,
    "Miou": 0.9600249517099162
}
Scores: {
    "AP": 0.9960450276187283,
    "mAP": 0.9958018281091826,
    "Miou": 0.9628756129265168
}
Scores: {
    "AP": 0.9957224990151919,
    "mAP": 0.9952550940049812,
    "Miou": 0.9588138290067088
}
Scores: {
    "AP": 0.9954449281103313,
    "mAP": 0.9950237741626029,
    "Miou": 0.9577963724944052
}
Scores: {
    "AP": 0.9954534057173884,
    "mAP": 0.9951255890176892,
    "Miou": 0.9576458433426079
}
Scores: {
    "AP": 0.9955198126798358,
    "mAP": 0.9952347384361282,
    "Miou": 0.9561153106668286
}
Scores: {
    "AP": 0.9949897226850516,
    "mAP": 0.9949172539971273,
    "Miou": 0.954747847430041
}
Scores: {
    "AP": 0.9948806517554738,
    "mAP": 0.9948466920388258,
    "Miou": 0.9535655598587435
}
Scores: {
    "AP": 0.9949384506233905,
    "mAP": 0.9949157733890949,
    "Miou": 0.9526716113221664
}
Scores: {
    "AP": 0.9945924617685366,
    "mAP": 0.9945584114567602,
    "Miou": 0.9520913999803904
}
Scores: {
    "AP": 0.9947179413008159,
    "mAP": 0.9946988217052838,
    "Miou": 0.9521903700310705
}
Scores: {
    "AP": 0.9946939382145614,
    "mAP": 0.9946856536027961,
    "Miou": 0.9522700110461394
}
Scores: {
    "AP": 0.994691711769917,
    "mAP": 0.9947315027035891,
    "Miou": 0.9520576129052454
}
Scores: {
    "AP": 0.9949507209743795,
    "mAP": 0.9949117485936476,
    "Miou": 0.9521261155941376
}
Scores: {
    "AP": 0.9946956614095608,
    "mAP": 0.9947309443459644,
    "Miou": 0.9529336142366911
}
Scores: {
    "AP": 0.9945577596972167,
    "mAP": 0.994472820717184,
    "Miou": 0.9514196463300055
}
Scores: {
    "AP": 0.9946079201377943,
    "mAP": 0.9945840722349204,
    "Miou": 0.9516844659075777
}
Scores: {
    "AP": 0.9945341605292095,
    "mAP": 0.9944855813144003,
    "Miou": 0.9519562462202273
}
Scores: {
    "AP": 0.9948991057164015,
    "mAP": 0.9946369169183755,
    "Miou": 0.9530618853546317
}
Scores: {
    "AP": 0.9949558961047006,
    "mAP": 0.9946696504045873,
    "Miou": 0.9532518616590787
}
Scores: {
    "AP": 0.9948732248661021,
    "mAP": 0.9945326699389337,
    "Miou": 0.9525236806356244
}
Scores: {
    "AP": 0.9947225380455669,
    "mAP": 0.9943869783209753,
    "Miou": 0.9520368785788115
}
Scores: {
    "AP": 0.9947707157256257,
    "mAP": 0.9944944726284296,
    "Miou": 0.9516062310199793
}
Scores: {
    "AP": 0.9947502276254676,
    "mAP": 0.9944747505318056,
    "Miou": 0.9518687911266152
}
Scores: {
    "AP": 0.9944861365959483,
    "mAP": 0.9942497840962073,
    "Miou": 0.9519145315200347
}
Scores: {
    "AP": 0.9944528659510138,
    "mAP": 0.9942404695849085,
    "Miou": 0.9523282758810867
}
Scores: {
    "AP": 0.9942988809308871,
    "mAP": 0.9940521342890485,
    "Miou": 0.9519792734528323
}
Scores: {
    "AP": 0.9943415669365948,
    "mAP": 0.994113626313884,
    "Miou": 0.952422997917445
}
Scores: {
    "AP": 0.9943450026246226,
    "mAP": 0.9941146958882023,
    "Miou": 0.9523039487422166
}
Scores: {
    "AP": 0.9943226253685628,
    "mAP": 0.9941329720318026,
    "Miou": 0.9515888932793891
}
Scores: {
    "AP": 0.9942592485092501,
    "mAP": 0.9940932989446483,
    "Miou": 0.951336019940859
}
Scores: {
    "AP": 0.9942922015659373,
    "mAP": 0.9941228471079666,
    "Miou": 0.9511451578662118
}
Scores: {
    "AP": 0.9940085050594083,
    "mAP": 0.9938768533762167,
    "Miou": 0.9504551320876777
}
Scores: {
    "AP": 0.993997880586797,
    "mAP": 0.9938752536884012,
    "Miou": 0.9504148807267779
}
Scores: {
    "AP": 0.994077700192807,
    "mAP": 0.9939544020683213,
    "Miou": 0.9509176682315418
}
Scores: {
    "AP": 0.9940495563402056,
    "mAP": 0.9938722606894131,
    "Miou": 0.9503078287574311
}
Scores: {
    "AP": 0.9940912954031362,
    "mAP": 0.9939495092121734,
    "Miou": 0.9505371410614084
}
Scores: {
    "AP": 0.9940262293628878,
    "mAP": 0.9939268856811253,
    "Miou": 0.9503528372070955
}
Scores: {
    "AP": 0.9938301516281416,
    "mAP": 0.9938191492505458,
    "Miou": 0.9498354790331176
}
Scores: {
    "AP": 0.9938659772333835,
    "mAP": 0.9938773901990279,
    "Miou": 0.9497326403344558
}
Scores: {
    "AP": 0.993961329647621,
    "mAP": 0.9939998570647939,
    "Miou": 0.9505219692202137
}
Scores: {
    "AP": 0.9939575977733811,
    "mAP": 0.9939953742469315,
    "Miou": 0.9507083017330972
}
Scores: {
    "AP": 0.9940056367100727,
    "mAP": 0.9940207071698062,
    "Miou": 0.9505859815603672
}
Scores: {
    "AP": 0.9940076906473196,
    "mAP": 0.9940252787513231,
    "Miou": 0.9510460951420938
}
Scores: {
    "AP": 0.9940501062982675,
    "mAP": 0.9940694201822353,
    "Miou": 0.9513665668117371
}
Scores: {
    "AP": 0.9939698187960548,
    "mAP": 0.9939717156953942,
    "Miou": 0.9509987892342673
}
Scores: {
    "AP": 0.9939961966091715,
    "mAP": 0.9940061900960968,
    "Miou": 0.9512862174120456
}
Scores: {
    "AP": 0.9939070880543456,
    "mAP": 0.9938913157313732,
    "Miou": 0.951057110349099
}
Scores: {
    "AP": 0.9939246028201665,
    "mAP": 0.993913357425547,
    "Miou": 0.9509548932856978
}
Scores: {
    "AP": 0.9938150870251602,
    "mAP": 0.9938298616007731,
    "Miou": 0.9507541557173418
}
Scores: {
    "AP": 0.9939621104395092,
    "mAP": 0.9938884918028407,
    "Miou": 0.9510639303365606
}
Scores: {
    "AP": 0.9939819997433099,
    "mAP": 0.9939291226923809,
    "Miou": 0.9506148206879697
}
Scores: {
    "AP": 0.9939282808768339,
    "mAP": 0.9938950205116742,
    "Miou": 0.950701075196016
}
Scores: {
    "AP": 0.993794717781581,
    "mAP": 0.9938156192865949,
    "Miou": 0.9504651033589099
}
Scores: {
    "AP": 0.9937781587530897,
    "mAP": 0.9938295093318466,
    "Miou": 0.9504628536020754
}
Scores: {
    "AP": 0.9938208446820881,
    "mAP": 0.9938763283736652,
    "Miou": 0.9501350988354553
}
Scores: {
    "AP": 0.9939405798314779,
    "mAP": 0.9939434032430953,
    "Miou": 0.9502291263822045
}
```
