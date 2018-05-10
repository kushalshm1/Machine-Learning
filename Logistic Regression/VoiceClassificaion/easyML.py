import numpy as np 
import pandas as pd

# class logisitcRegressionClass:
    # mainData      = pd.DataFrame(pd.read_csv(data))
    # mainData      = mainData.dropna(axis=0, how='any')
    # mainLabel     = mainData[label]
    # uniqueLabels  = list(set(mainLabel))
    # train,cv,test = np.split(mainData,[int(0.6*len(mainData)),int(0.8*len(mainData))])
    # trainFeatures = train.select_dtypes(include=['float64','int64'])
    # trainLabel    = train[label]
    # trainFeatures = trainFeatures.T
    # testFeatures  = train.select_dtypes(include=['float64','int64'])
    # testLabel     = test[label]
    # testFeatures  = testFeatures.T
    # cvFeatures    = train.select_dtypes(include=['float64','int64'])
    # cvLabel       = cv[label]
    # cvFeatures    = cvFeatures.T
def logisticRegressionTrain(data,iters,lr):
    mainData      = pd.DataFrame(pd.read_csv("voice.csv"))

    # Selecting Label Vector
    tempData = mainData
    tempData = tempData.T
    tempUniqueSet = []
    temp = []
    for i in range(0,len(tempData)):
        tempUniqueSet.append(list(set(tempData.iloc[i])))
    # print(tempUniqueSet)
    x = (min(tempUniqueSet,key=len))
    labelIndex = tempUniqueSet.index(x)
    label = mainData.columns[labelIndex]
    ########################

    mainData      = mainData.sample(frac=1).reset_index(drop=True)
    mainData      = mainData.dropna(axis=0, how='any')
    mainLabel     = mainData[label]
    uniqueLabels  = list(set(mainLabel))
    train,cv,test = np.split(mainData,[int(0.6*len(mainData)),int(0.8*len(mainData))])
    trainFeatures = train.select_dtypes(include=['float64','int64'])
    trainLabel    = train[label]
    trainFeatures = trainFeatures.T
    testFeatures  = train.select_dtypes(include=['float64','int64'])
    testLabel     = test[label]
    testFeatures  = testFeatures.T
    cvFeatures    = train.select_dtypes(include=['float64','int64'])
    cvLabel       = cv[label]
    cvFeatures    = cvFeatures.T

    '''Changing Labels'''
    trainLabel = trainLabel.replace(uniqueLabels,[0,1])
    testLabel  = testLabel.replace(uniqueLabels,[0,1])
    cvLabel    = cvLabel.replace(['male','female'],[0,1])

    featureCount = len(trainFeatures)
    weights      = [0.1]*featureCount

    def sigmoid(weights,features,label):

        z = 0
        for i in range(1,len(weights)):
            z = z + weights[i]*features.iloc[i]
        z = z+weights[0]
        return 1/(1+np.exp(-z))

    def cost(weights,features,label):
        predicted = sigmoid(weights,features,label)
        costClass1 = -label*np.log(predicted)
        costClass2 = (1-label)*np.log(1-predicted)
        cost = costClass1-costClass2
        cost = np.mean(cost)
        return cost



    def derivative(weights,features,label,featureIteration):
        predictions = sigmoid(weights,features,label)
        derivative = np.dot((features.iloc[featureIteration]).T,  predictions - label)
        derivative = np.mean(derivative)
        return derivative


    def gradientDescent(weights,features,label,iters,lr):
        weightsNew = [0]*len(weights)
        for i in range(0,iters):
            for n in range(0,len(weights)):
                weightsNew[n] = weights[n] - lr*derivative(weights,features,label,n)
            weights = weightsNew
        return weights

    '''Final Training Area'''
    trainedWeights = (gradientDescent(weights,trainFeatures,trainLabel,iters,lr))
    trainedWeights = gradientDescent(weights,trainFeatures,trainLabel,iters,lr)
    # from sklearn.metrics import accuracy_score
    predicted = sigmoid(trainedWeights,trainFeatures,trainLabel)
    for i in range(0,len(predicted)):
        if predicted.iloc[i]>0.5:
            predicted.iloc[i] = 1
        else:
             predicted.iloc[i] = 0
    predicted = list(map(int, predicted))
            
    # accuracy = (accuracy_score(trainLabel,predicted)*100)
    correct = 0
    for i in range(len(predicted)):
        if(predicted[i]==trainLabel.iloc[i]):
            correct = correct+1

    print("=================================================")
    print("---------------Training Accuracy-----------------")
    # print("               ",accuracy,"%","                  ")
    print("Correct Answers:",correct,"Out of:",len(trainLabel))
    print("=================================================")  

    return trainedWeights

def logisticRegressionTest(weights,features):
    sig = 0
    weights = np.array(weights)
    features = np.array(features)
    z = 0
    z = np.dot(weights,features)
    sig = 1/(1+np.exp(-z))
    if(sig>0.5):
        print("Male")
    else:
        print("Female")


