---
title: lgbm을 이용한 예측 알고리즘
---

#### 개요

Dacon code 공유에 있는 baseline code를 리뷰해보록 한다. 본 알고리즘은 아래에 있는 코드를 그대로 가져온 것이다. https://dacon.io/competitions/official/235713/codeshare/2476?page=2&dtype=recent
아직 부족한 데이터프레임에 대한 이해와 다양한 데이터를 통해 어떻게 정확한 예측값을 얻을 수 있는지 공부한다.


```
import pandas as pd
import numpy as np
import warnings #경고 필터
import zipfile as zip
warnings.filterwarnings("ignore")

import glob
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
import random 
```


기본적으로 Dataframe의 기초가 되는 pandas , numpy 그리고 데이터 전처리를 위한 OneHotEncoder, zip , glob 마지막으로 학습을 
위한 LGBMClassifier , StratifiedKFold(사실 애는 전처리에 들어가야하나?) 모듈들을 import해주었다.


```
zip_r = zip.ZipFile("/content/drive/MyDrive/My project/dataset/open (1).zip");
zip_r.extractall("/content/drive/MyDrive/My project/dataset")
zip_r.close()

train =pd.read_csv("/content/drive/MyDrive/My project/dataset/open/train.csv")
test = pd.read_csv("/content/drive/MyDrive/My project/dataset/open/test.csv")
sample_submission = pd.read_csv("/content/drive/MyDrive/My project/dataset/open/sample_submission.csv")

train.shape
```


zip 모듈을 통해 데이터를 가져와 pd.readcsv메소드로 csv파일을 불러온다. train.shape는 (26457, 20) 가 나온다.


```
train.isnull().sum()


결과
index               0
gender              0
car                 0
reality             0
child_num           0
income_total        0
income_type         0
edu_type            0
family_type         0
house_type          0
DAYS_BIRTH          0
DAYS_EMPLOYED       0
FLAG_MOBIL          0
work_phone          0
phone               0
email               0
occyp_type       8171
family_size         0
begin_month         0
credit              0
dtype: int64
```

데이터에서 결측값을 나타내었다. 결측값은 학습에 있어 한두개 면 몰라도 8000개 가량의 데이터 손실은 학습에 않좋은 영향을 미칠
것이다.. 따라서 삭제해준다.


```
train.drop(["index"] , axis = 1)
train.fillna("NAN" , inplace = True);#결측값들을 알아서 채워줌

test.drop(["index"] , axis =1)
test.fillna("NAN"  , inplace=True)

train.isnull().sum()# 결측값 사라짐


결과
gender           0
car              0
reality          0
child_num        0
income_total     0
income_type      0
edu_type         0
family_type      0
house_type       0
DAYS_BIRTH       0
DAYS_EMPLOYED    0
FLAG_MOBIL       0
work_phone       0
phone            0
email            0
occyp_type       0
family_size      0
begin_month      0
credit           0
dtype: int64
````

학습에 있어 의미가 없는 index 컬럼을 삭제하고 dataframe의 fillna메소드로 결측값에 "NAN"을 채워주어 여기서 inplace는 디폴트 값인 inplace = False는
원본 dataframe은 직접적으로 drop변경하지 않으며 원본 dataframe에서 데이터가 drop된 새로운 dataframe을 return 반환 해준다. 하지만 
inplace = true 이면 메모리를 copy하여 삭제하지 않고 원본 dataframe의 메모리에 그대로 적용이 된다. 그리고 반환은 None을 해준다.


```
object_col = []
for col in train.columns:
  if train[col].dtype == 'object':
    object_col.append(col)
    
print(object_col)


결과
['gender', 'car', 'reality', 'income_type', 'edu_type', 'family_type', 'house_type', 'occyp_type']
```

컬럼의 데이터 타입이 object즉 one hot encodr 를 통해 변환가능한 컬럼값들만 추출한다,


```
enc  = OneHotEncoder();
enc.fit(train.loc[: , object_col])
train_onehot_df = pd.DataFrame(enc.transform(train.loc[: , object_col]).toarray() , 
                               columns = enc.get_feature_names(object_col))
train.drop(object_col , axis=1 , inplace= True)
train = pd.concat([train , train_onehot_df] , axis = 1)
```

one hot encoding 단하나의 값만 true이고 나머지느 모두 false로 만든다. Encode를 생성후 fit& transform 함수를 호출한다 fit메소드를 
호출 할때 분류에 사용되는 클래스들을 식별하고 메타를 기록해둔다. transform함수를 호출할 때 One-Hot 인코딩된 결과를 리턴란다. 만약 fit
호출과정에서 보지 못한 컬럼이 transform호출시 나타나면 오류메세지를 발생 시킨다.

따라서 train_onehot_df에는 인코딩한 값들이 새로운 데이터 프레임에 들어가게되고 데이터 타입이 object였던 컬럼들은 drop을 통해 삭제 해준다. 삭제해준 컬럼대신에 one-hot-encoding한 dataframe을 concat해준다. 

```
enc.fit(test.loc[: , object_col])
test_onehot_df = pd.DataFrame(enc.transform(test.loc[: , object_col]).toarray() , 
                              columns = enc.get_feature_names(object_col))
test = test.drop(object_col , axis =1 , inplace=True)
test = pd.concat([test,  test_onehot_df] , axis = 1)
```

test데이터도 마찬가지이다 objecr value를 가지는 컬럼들을 삭제하고 encoding된 daafarame으로 합해 준다.


```
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)#교차검증을 위한 전처리
folds=[]
for train_idx, valid_idx in skf.split(train, train['credit']):
    folds.append((train_idx, valid_idx))
```
stratifiedKFold 라는 메소드를 찾아보면 F-Folds cross-validator을 찾을 수 있다 설명으로는 train/test 데이터를 나누어 제공한다는 것인데 여기서 n_splits는 number of folds 이다 최소 2 이상이 되어야하는데 데이터를 n만큼 구분한후 교차로 검증방식을 통해 학습률을 높인다. n값의 폴드세트에 n번의 학습과 검증을 준다. 

한번의 학습을 통해 평가 할 경우 과적합이 일어날 가능성이 있으므로 교차 검증을 통해 과적합을 막아준다.


```
random.seed(42)
lgb_models={}
for fold in range(5):
    print(f'===================================={fold+1}============================================')
    train_idx, valid_idx = folds[fold]
    X_train, X_valid, y_train, y_valid = train.drop(['credit'],axis=1).iloc[train_idx].values, train.drop(['credit'],axis=1).iloc[valid_idx].values,\
                                         train['credit'][train_idx].values, train['credit'][valid_idx].values 
    lgb = LGBMClassifier(n_estimators=1000)
    lgb.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_valid, y_valid)], 
            early_stopping_rounds=30,
           verbose=100)
    lgb_models[fold]=lgb
    print(f'================================================================================\n\n')
  ```
  학습을 하는 코드이다 .LGMBclassifer을 두고 학습을 해준다. 이떄 파라미터 튜닝을 해주면 학습률이 더욱 높아질 것이다.
  









