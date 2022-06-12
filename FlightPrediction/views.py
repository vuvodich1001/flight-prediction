from django.http import JsonResponse
import numpy as np 
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.linear_model import LinearRegression


def getValueOfJSON(json_data):
    json_load = (json.loads(json_data))
    dataTestArr = [] 
    for x in json_load:
        dataTestArr.append(json_load[x])
    return dataTestArr

def insert(dataframe, row):
    dataframe.loc[-1] = row  # adding a row
    dataframe.index = dataframe.index + 1  # shifting index
    dataframe = dataframe.sort_index()
    return dataframe

class DataTest:
  def __init__(self, airline_AirAsia, airline_Air_India, airline_GO_FIRST, airline_Indigo, airline_SpiceJet, airline_Vistara):
    self.duration = 2
    self.days_left = 10
    self.airline_AirAsia = airline_AirAsia
    self.airline_Air_India = airline_Air_India
    self.airline_GO_FIRST = airline_GO_FIRST
    self.airline_Indigo = airline_Indigo
    self.airline_SpiceJet = airline_SpiceJet
    self.airline_Vistara = airline_Vistara
    self.source_city_Bangalore = 0
    self.source_city_Chennai = 0
    self.source_city_Delhi = 1
    self.source_city_Hyderabad = 0
    self.source_city_Kolkata = 0
    self.source_city_Mumbai = 0
    self.stops_one = 0
    self.stops_two_or_more = 0
    self.stops_zero = 1
    self.destination_city_Bangalore = 0
    self.destination_city_Chennai = 0
    self.destination_city_Delhi = 0
    self.destination_city_Hyderabad = 0
    self.destination_city_Kolkata = 0
    self.destination_city_Mumbai = 1
    self.class_Business = 0
    self.class_Economy = 1

def sortSecond(val):
    return val[1]

def getData(request):
    df = pd.read_csv('FlightPrediction\Clean_Dataset.csv')
    df = df.drop(['Unnamed: 0', 'arrival_time', 'departure_time'] , axis='columns')
    # Tách thành 2 phần: 1 phần là thuộc tính bình thường, 1 phần là thuộc tính quyết định(nhãn)
    features = df.drop(['price'], axis = 1)
    labels = df['price']

    features.select_dtypes(exclude = ['int64', 'float64']).columns

    # Chuyển các cột có dữ liệu dạng chuỗi thành dạng binary 0, 1
    features_onehot = pd.get_dummies(features, columns = features.select_dtypes(exclude = ['int64', 'float64']).columns)

    # chia du lieu thanh 2 phan: 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(features_onehot, labels, test_size = 0.3, random_state = 42)

    # Khởi tạo đối tượng linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Datatrain 
    dataTrain = X_test.sort_index()

    
    dataTest1 = DataTest(0, 0, 0, 0, 0, 1) # Vistara
    dataTest2 = DataTest(0, 0, 0, 0, 1, 0) # SpiceJet
    dataTest3 = DataTest(0, 0, 0, 1, 0, 0) # Indigo
    dataTest4 = DataTest(0, 0, 1, 0, 0, 0) # GO_FIRST
    dataTest5 = DataTest(0, 1, 0, 0, 0, 0) # Air_India
    dataTest6 = DataTest(1, 0, 0, 0, 0, 0) # AirAsia

    jsonStr1 = json.dumps(dataTest1.__dict__)
    jsonStr2 = json.dumps(dataTest2.__dict__)
    jsonStr3 = json.dumps(dataTest3.__dict__)
    jsonStr4 = json.dumps(dataTest4.__dict__)
    jsonStr5 = json.dumps(dataTest5.__dict__)
    jsonStr6 = json.dumps(dataTest6.__dict__)

    dataTrain = insert(dataTrain, getValueOfJSON(jsonStr1))
    dataTrain = insert(dataTrain, getValueOfJSON(jsonStr2))
    dataTrain = insert(dataTrain, getValueOfJSON(jsonStr3))
    dataTrain = insert(dataTrain, getValueOfJSON(jsonStr4))
    dataTrain = insert(dataTrain, getValueOfJSON(jsonStr5))
    dataTrain = insert(dataTrain, getValueOfJSON(jsonStr6))

    # Test
    result = []
    i = 2
    for x in range(6):
        result.append([dataTrain.columns[i], abs(model.predict(dataTrain)[x])])
        i+=1
    result = result.sort(key=sortSecond)
    return JsonResponse(
        {
            'name':'python django',
            'length': json.dumps(result)
        }
        )