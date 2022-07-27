import pandas as pd

df = pd.read_csv('train.csv')
print(df.info())

"""
 #   Column            Non-Null Count  Dtype
---  ------            --------------  -----
 0   id                8193 non-null   int64 
 1   sex               8193 non-null   int64 
 2   bdate             7025 non-null   object 
 3   has_photo         8193 non-null   int64 
 4   has_mobile        8193 non-null   float64 
 5   followers_count   8193 non-null   float64 
 6   graduation        8193 non-null   float64 
 7   education_form    7585 non-null   object 
 8   relation          8193 non-null   float64 
 9   education_status  8193 non-null   object 
 10  langs             8193 non-null   object 
 11  life_main         8193 non-null   object 
 12  people_main       8193 non-null   object 
 13  city              7611 non-null   object 
 14  last_seen         8193 non-null   object 
 15  occupation_type   8000 non-null   object 
 16  occupation_name   8000 non-null   object 
 17  career_start      8193 non-null   object 
 18  career_end        8193 non-null   object 
 19  result            8193 non-null   int64 
"""

df['occupation_name'].fillna(-1, inplace=True)
df['occupation_type'].fillna(-1, inplace=True)
df['education_form'].fillna(-1, inplace=True)
df['bdate'].fillna(-1, inplace=True)
df['city'].fillna(-1, inplace=True)
df['last_seen'].fillna(-1, inplace=True)

print(len(df[pd.isnull(df['occupation_name'])]))
print(len(df[pd.isnull(df['occupation_type'])]))
print(len(df[pd.isnull(df['education_form'])]))
print(len(df[pd.isnull(df['bdate'])]))
print(len(df[pd.isnull(df['city'])]))
print(len(df[pd.isnull(df['last_seen'])]))

def occupation_format(data):
    if data == -1:
        return data
    elif "university" in data:
        return 1
    elif "work" in data:
        return 2

df['occupation_type'] = df['occupation_type'].apply(occupation_format)
print(df["occupation_type"].value_counts())

def education_format(data):
    if data == -1:
        return data
    elif "Full-time" in data:
        return 1
    elif "Distance Learning" in data:
        return 2
    elif "Part-time" in data:
        return 3

df['education_form'] = df['education_form'].apply(education_format)
print(df["education_form"].value_counts())

def people_format(data):
    if "False" in data:
        return 7
    else:
        return int(data)

df['people_main'] = df['people_main'].apply(people_format)
print(df["people_main"].value_counts())

def life_main_format(data):
    if "False" in data:
        return 9
    else:
        return int(data)

df['life_main'] = df['life_main'].apply(life_main_format)
print(df["life_main"].value_counts())

def career_start_format(data):
    if "False" in data:
       return 1
    else:
        return int(data)

df['career_start'] = df['career_start'].apply(career_start_format)
print(df["career_start"].value_counts())

def career_end_format(data):
    if "False" in data:
       return 1
    else:
        return int(data)

df['career_end'] = df['career_end'].apply(career_end_format)
print(df["career_end"].value_counts())

def education_status_format(data):
    if "Alumnus (Specialist)" in data:
        return 1
    elif "Student (Specialist)" in data:
        return 2
    elif "Student (Bachelor's)" in data:
        return 3
    elif "Alumnus (Bachelor's)" in data:
        return 4
    elif "Alumnus (Master's)" in data:
        return 5
    elif "PhD" in data:
        return 6
    elif "Student (Master's)" in data:
        return 7
    elif "Undergraduate applicant" in data:
        return 8
    elif "Candidate of Sciences" in data:
        return 9
        
df['education_status'] = df['education_status'].apply(education_status_format)
print(df["education_status"].value_counts())

print(df.info())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

i = 0
aver_result = 0

df = df.drop('id', axis = 1)
df = df.drop('bdate', axis = 1)
df = df.drop('followers_count', axis = 1)
df = df.drop('langs', axis = 1)
df = df.drop('city', axis = 1)
df = df.drop('last_seen', axis = 1)
df = df.drop('occupation_name', axis = 1)
df = df.drop('career_start', axis = 1)
df = df.drop('career_end', axis = 1)
df = df.drop('graduation', axis = 1)

def convert(data):
    if type(data)==str():
        return float(data)

while i<5:
    x = df.drop('result', axis = 1)
    y = df['result']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    classifier = KNeighborsClassifier(n_neighbors = 5)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    print('Процент правильно предсказанных исходов:', accuracy_score(y_test, y_pred) * 100)
    aver_result += accuracy_score(y_test, y_pred) * 100
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))
    i += 1
print('Процент правильно предсказанных исходов:', aver_result/5)