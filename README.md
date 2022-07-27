
# FirstNeuralNetwork

Это моя первая нейронная сеть, которую я написал самостоятельно и вручную.
Весь код был написан на Python.

## Ради чего нужна эта математическая модель?
*Эта математическая модель прогнозирует покупки тех кто купил и тех кто не купил курсы на сайте*

*Прогноз составлен на основе данных с датасета "train.csv"*

**Ссылка на датасет: https://github.com/Kameton111/images/blob/main/train.csv**

## Благодаря каким данным мы прогназируем?

*Данные пользователей, которые были использованы:*

1. Текущее занятие пользователя (школа, университет, работа). 
1. Форма обучения.
1. Главное в людях. (1 — ум и креативность;  2 — доброта и честность; 3 — красота и здоровье;  4 — власть и богатство; 5 — смелость и упорство; 6 — юмор и жизнелюбие).
1. Год начала работы.
1. Год окончания работы.
1. Статус обучения.

## Структура модели

*Сперва устанавливаем библиотеку sklearn.*

**Sklearn - это библиотека для машинного обучения на языке программирования Python.**

*Потом импортируем все нужные элементы*


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

*Создаем две переменные равные нулю*

i = 0
aver_result = 0

*Затем удаляем ненужные столбцы:*

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

*Главная часть математической модели:*

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


## Результат работы модели:

![](https://github.com/Kameton111/images/blob/main/digital_edu.py%20-%20level%20(Workspace)%20-%20Visual%20Studio%20Code%202022-07-25%2020-16-22.gif)

*После активации модели в терминале показывается вся информация о датафрейме и пять различных прогнозов*

## Библиотеки, которые были использованы:

*1. Библиотека для анализа данных Pandas*

https://pandas.pydata.org/ *оффициальный сайт Pandas*

*2. Библиотека машинного обучения Sklearn*

https://scikit-learn.org/stable/ *оффициальный сайт Sklearn*
## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


## API Reference

#### Get all items

```http
  GET /api/items
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `api_key` | `string` | **Required**. Your API key |

#### Get item

```http
  GET /api/items/${id}
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `id`      | `string` | **Required**. Id of item to fetch |

#### add(num1, num2)

Takes two numbers and returns the sum.


## Authors

- [@katherinepeterson](https://www.github.com/octokatherine)


## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)

## Color Reference

| Color             | Hex                                                                |
| ----------------- | ------------------------------------------------------------------ |
| Example Color | ![#0a192f](https://via.placeholder.com/10/0a192f?text=+) #0a192f |
| Example Color | ![#f8f8f8](https://via.placeholder.com/10/f8f8f8?text=+) #f8f8f8 |
| Example Color | ![#00b48a](https://via.placeholder.com/10/00b48a?text=+) #00b48a |
| Example Color | ![#00d1a0](https://via.placeholder.com/10/00b48a?text=+) #00d1a0 |


## Documentation

[Documentation](https://linktodocumentation)


## Installation

Install my-project with npm

```bash
  npm install my-project
  cd my-project
```
    
## 🚀 About Me
I'm a full stack developer...


## Feedback

If you have any feedback, please reach out to us at fake@fake.com


## Demo

Insert gif or link to demo


## FAQ

#### Question 1

Answer 1

#### Question 2

Answer 2


## Appendix

Any additional information goes here

