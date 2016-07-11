#!/usr/bin/env python

import sys
import os
from numpy import *
import math
import numpy as np
import operator
import pickle
#import MLLIB_1

MIN_FLOAT_VALUE = 1e-6

def classify_knn(inX, dataSet, labels, k):   #dataSet--numpy array:2-dim; inX,labels--list
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2  #numpy array do not use matrix multiply
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


'''
Created on Aug 20, 2015
N-Bayes: naive bayes : <<fangfa>>P51 

        feature: set and numeric(discrete, and if continuous, need to quantize) 
        labels: set and numeric (discrete, and if continuous, need to quantize)

Input:      dataSet: size n data set of known vectors (M*N), discrete, feature j has Sj values
            labels: data set labels (1xN vector),  discrete, label has K valuses
            lamda: laplace smoothing default 1
            
Output:     the possibility of p(y=ck) and p(x=xj|y=ck)

@author: xch 
'''

def quantize(inData,quantize_interval):   #inData--numpy array,N*M,M features:1(array(1-dim list), zeros(N)) or 2 dim, quantize_interval--list
    outData = inData.copy()
    print inData.shape
    if len(inData.shape) == 1:
        quantize_nums = int((max(inData) - min(inData))/quantize_interval[0]) + 1
        q = [0] * quantize_nums
        for i in range(quantize_nums):
            q[i] = min(inData) + i*quantize_interval[0]
        for i in range(len(inData.shape[0])):
            for j in range(quantize_nums):
                if j < quantize_nums - 1:
                    if inData[i] >= q[j] and inData[i] < q[j+1]:
                        outData[i] = q[j] 
                        break
                if j == quantize_nums - 1:
                    if inData[i] >= q[j]:
                        outData[i] = q[j] 
                        break
                    else:
                        print "quantize wrong"
    else:
        for feature in range(inData.shape[1]):
            quantize_nums = int((max(inData[:,feature]) - min(inData[:,feature]))/quantize_interval[feature]) + 1
            q = [0] * quantize_nums
            for i in range(quantize_nums):
                q[i] = min(inData[:,feature]) + i*quantize_interval[feature]
            for i in range(inData.shape[0]):
                for j in range(quantize_nums):
                    if j < quantize_nums - 1:
                        if inData[i,feature] >= q[j] and inData[i,feature] < q[j+1]:
                            outData[i,feature] = q[j] 
                            break
                    if j == quantize_nums - 1:
                        if inData[i,feature] >= q[j]:
                            outData[i,feature] = q[j] 
                            break
                        else:
                            print "quantize wrong"
                
    return outData 

def quantize_match(inData, inX):    #inData--numpy array: 2-dim, inX--list
    outX = [0] * len(inX) 
    if not inData.shape[1] == len(inX):
        print "quantize_match wrong"
        os.exit(-1) 
    for f in range(inData.shape[1]):
        MIN_VALUE = 65535
        min_index = -1
        for n in range(inData.shape[0]):
            if abs(inX[f] - inData[n,f]) < MIN_VALUE:
                MIN_VALUE = abs(inX[f] - inData[n,f])
                min_index = n
        outX[f] = inData[min_index, f]


    return outX

def build_bayes(dataSet, labels, lamda=1.0):  #dataSet--numpy array:2-dim,discrete; labels--list,discrete; lamda--laplace smoothing
    dataSetSizeN = dataSet.shape[0] #N
    dataSetSizeM = dataSet.shape[1] #M
    K = len(list(set(labels)))
    S = [0]*dataSetSizeM 
    for j in range(dataSetSizeM):
        S[j] = len(list(set(list(dataSet[:,j])))) # note: 2-dim array can not be to list, 2 dim list can be to array

    Py = {} 
    Px_y = [] 
    labels_numpy_array = array(labels)
    for ck in list(set(labels)):
        Py[ck] = float(len(labels_numpy_array[labels_numpy_array == ck]) + lamda)/float(dataSetSizeN + K*lamda) 
    for j in range(dataSetSizeM):
        apple = {}
        feature_j_numpy_array = dataSet[:,j]
        for ck in list(set(labels)):
            for xj in list(set(list(dataSet[:,j]))):
                joint_possibility = 0
                for xj_index in list(where(feature_j_numpy_array == xj)[0]):
                    if labels[xj_index] == ck:
                        joint_possibility += 1
                apple[(xj,ck)] = float(joint_possibility + lamda)/float(len(labels_numpy_array[labels_numpy_array == ck]) + S[j]*lamda)
        Px_y.append(apple)
            
    return Px_y, Py 



def classify_bayes(inX, px_y, py):   #inX--list, 1*M, M features
    class_y = {}
    for ck in py.keys():
        class_y[ck] = 1
        for j in range(len(inX)):
            if not px_y[j].has_key((inX[j],ck)):
                print j,inX[j],ck
            class_y[ck] = class_y[ck] * px_y[j][(inX[j],ck)]
        class_y[ck] = class_y[ck] * py[ck]
    sortedClass_y= sorted(class_y.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClass_y[0][0]




'''
cart tree:
        feature: numeric (discrete or continuous)
        labels: numeric (regLeaf, modelLeaf) and set(Gini)
external:   
            createTree()
            prune()
            createForeCast()
            
@author: xch 
'''

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    return mat0, mat1

def regLeaf(dataSet):
    return mean(dataSet[:,-1])

def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def Gini(dataSet):
    C_list_tmp = list(set(dataSet[:,-1].T.tolist()[0]))
    C_list = [int(x) for x in C_list_tmp]
    #print C_list
    K = len(C_list)
    D_abs = shape(dataSet)[0]
    Ck_abs = [0]*K
    for Ck_index in range(len(C_list)):
        for i in range(D_abs):
            if int(dataSet[i,-1]) == C_list[Ck_index]:
                #print C_list[Ck_index]
                Ck_abs[Ck_index] += 1
    #print Ck_abs
    return 1-sum([float(x*x) for x in Ck_abs])/pow(float(D_abs),2)
    #return 1-sum(pow(float(Ck_abs),2))/pow(float(D_abs),2)

def GiniLeaf(dataSet):
    return [int(x) for x in list(set(dataSet[:,-1].T.tolist()[0]))]

    

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(0,1)):
    tolS = ops[0]; tolN=ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
        #for splitVal in set(dataSet[:,featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            if errType == regErr:
                newS = errType(mat0) + errType(mat1)
            elif errType == Gini:
                newS = float(shape(mat0)[0])/float(shape(dataSet)[0]) * errType(mat0) + float(shape(mat1)[0])/float(shape(dataSet)[0]) * errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue

def isTree(obj):
    return (type(obj).__name__ == 'dict')
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree)
    if not isTree(tree): 
	    print "The tree is a float"
	    print tree
	    return tree
   # print "debug"
   # print tree['right']
   # print tree['left']
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData,tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'], 2)) + \
                       sum(power(rSet[:,-1] - tree['right'], 2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean
        else: return tree
    else: return tree

def linearSolve(dataSet):
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]
    xTx = X.T*X
    print linalg.det(xTx)
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
                        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))
    


# note: numpy matrix[:,1] is different from numpy array[:,1], the fronter is matrix 2-dim, the latter is array 1-dim
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):  #dataSet -- numpy matrix including labels
    feat, val = chooseBestSplit(dataSet,leafType, errType, ops)
    if feat == None: return val
    retTree= {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

def regTreeEval(model, inDat):
    return float(model)
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1] = inDat
    return float(X*model)

def GiniTreeEval(model, inDat):
    return model

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[0,tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):  #testData -- numpy.matrix
    m = len(testData)
    if modelEval == GiniTreeEval:
        #print "i am in gini tree"
        yHat = {}
        for i in range(m):
            yHat[i] = treeForeCast(tree, mat(testData[i]), modelEval)
    else:
        yHat = mat(zeros((m,1)))
        for i in range(m):
            yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    #print yHat
    return yHat

##labels: set
#myMat = mat(dataSet[conf_name])
#myTree = regTrees.createTree(myMat, regTrees.GiniLeaf, regTrees.Gini, (TOLS,TOLN))
#filename = "../tree" + "_" + str(conf_name) + ".txt"
#fw = open(filename,'w')
#pickle.dump(myTree,fw) 
#fw.close()
#
#feature_vector = mat(sample)
#fr = open("../tree_" + str(last_conf) +".txt")
#myTree = pickle.load(fr)
#conf_vector = regTrees.createForeCast(myTree, feature_vector, regTrees.GiniTreeEval)[0]
#fr.close()
#
#
##labels: numeric
#myMat1 = mat(myDat)
#for i1 in range(len(unique_list_2G)):
#    print "the %d times to tree" % i1
#    train_test_Mat1 = hstack((myMat1[:,len(unique_list_2G):(len(unique_list_2G)+len(unique_list_5G))],myMat1[:,i1])) 
#    train_sample_nums1 = int(0.85*shape(train_test_Mat1)[0])
#    trainMat1 = train_test_Mat1[0:train_sample_nums1,:]
#    testingMat1 = train_test_Mat1[train_sample_nums1:shape(train_test_Mat1)[0],:]
#    myTree1_tmp = regTrees.createTree(trainMat1, regTrees.regLeaf, regTrees.regErr, (TOLS,TOLN))
#    regTrees.prune(myTree1_tmp, testingMat1)
#
#    filename = "tree_5G_2G" + "_" + str(i1) + ".txt"
#    fw = open(filename,'w')
#    pickle.dump(myTree1_tmp,fw) 
#    fw.close()
#
#    fr = open("area" + "_" + str(AREA) + "/" + "tree_two_modes/" + filename)
#    myTree1_test = pickle.load(fr)
#    testMat1 = trainMat1 
#    yHat = regTrees.createForeCast(myTree1_test, testMat1[:,0:len(unique_list_5G)], regTrees.regTreeEval)
#    a = corrcoef(yHat, testMat1[:,-1],rowvar = 0)[0,1]


'''
ID3 tree:
        feature: set 
        labels: set 
external:   
            createTree_id3()
            classify_id3()
            
@author: xch 
'''

#def createDataSet():
#    dataSet = [[1, 1, 'yes'],
#               [1, 1, 'yes'],
#               [1, 0, 'no'],
#               [0, 1, 'no'],
#               [0, 1, 'no']]
#    labels = ['no surfacing','flippers']
#    #change to discrete values
#    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * math.log(prob,2) #log base 2
    return shannonEnt
    
def splitDataSet_id3(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
def chooseBestFeatureToSplit_id3(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet_id3(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree_id3(dataSet,labels): #dataSet--list 2-dim, including labels; labels--list 1-dim, name like ['no surfacing','flippers']
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit_id3(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree_id3(splitDataSet_id3(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
def classify_id3(inputTree,featLabels,testVec): #featLabels--list 1-dim, name like ['no surfacing','flippers']; testVec--list 1-dim
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify_id3(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel


'''
Logistic Regression 

        feature: numeric 
        labels: set and numeric must be 1 or 0
external:   
        gradAscent_lr()
        stocGradAscent0_lr()
        stocGradAscent1_lr()
        classifyVector_lr()
'''
#def loadDataSet():
#    dataMat = []; labelMat = []
#    fr = open('testSet.txt')
#    for line in fr.readlines():
#        lineArr = line.strip().split()
#        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
#        labelMat.append(int(lineArr[2]))
#    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent_lr(dataMatIn, classLabels):  # dataMatIn--list 2-dim, adding [1] like [1.0, float(lineArr[0]), float(lineArr[1])]; classLabels--list 1-dim
    dataMatrix = mat(dataMatIn)             
    labelMat = mat(classLabels).transpose() 
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):              
        h = sigmoid(dataMatrix*weights)     
        error = (labelMat - h)              
        weights = weights + alpha * dataMatrix.transpose()* error 
    return weights          #weights -- matrix 2-dim, note: matrix 2-dim can be converted to array 2-dim and then using a[:,0] to array 1-dim

#def plotBestFit(weights):
#    import matplotlib.pyplot as plt
#    dataMat,labelMat=loadDataSet() # dataMatIn--list 2-dim, adding [1] like [1.0, float(lineArr[0]), float(lineArr[1])]; classLabels--list 1-dim
#    dataArr = array(dataMat)
#    n = shape(dataArr)[0] 
#    xcord1 = []; ycord1 = []
#    xcord2 = []; ycord2 = []
#    for i in range(n):
#        if int(labelMat[i])== 1:
#            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
#        else:
#            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
#    ax.scatter(xcord2, ycord2, s=30, c='green')
#    x = arange(-3.0, 3.0, 0.1)
#    y = (-weights[0]-weights[1]*x)/weights[2]
#    ax.plot(x, y)
#    plt.xlabel('X1'); plt.ylabel('X2');
#    plt.show()

def stocGradAscent0_lr(dataMatrix, classLabels): #dataMatrix -- array 2-dim, adding [1]; classLabels -- list 1-dim
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights # array 1-dim

def stocGradAscent1_lr(dataMatrix, classLabels, numIter=150): #dataMatrix -- array 2-dim, adding [1]; classLabels -- list 1-dim
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights # array 1-dim

def classifyVector_lr(inX, weights): #inX --array 1-dim, adding [1]; weights --array 1-dim
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

#def colicTest(): #lost adding [1]
#    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
#    trainingSet = []; trainingLabels = []
#    for line in frTrain.readlines():
#        currLine = line.strip().split('\t')
#        lineArr =[]
#        for i in range(21):
#            lineArr.append(float(currLine[i]))
#        trainingSet.append(lineArr)
#        trainingLabels.append(float(currLine[21]))
#    trainWeights = stocGradAscent1_lr(array(trainingSet), trainingLabels, 1000)
#    errorCount = 0; numTestVec = 0.0
#    for line in frTest.readlines():
#        numTestVec += 1.0
#        currLine = line.strip().split('\t')
#        lineArr =[]
#        for i in range(21):
#            lineArr.append(float(currLine[i]))
#        if int(classifyVector_lr(array(lineArr), trainWeights))!= int(currLine[21]):
#            errorCount += 1
#    errorRate = (float(errorCount)/numTestVec)
#    print "the error rate of this test is: %f" % errorRate
#    return errorRate


'''
SVM 

        feature: numeric,m(benchmark) * n(features)
        labels: set and numeric must be 1 or -1 
external:   
        smoP()
        testRbf() 
'''

def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj


def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('That Kernel is not recognized')
    return K

class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
        
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek
        
def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
        
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:  return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0:  return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):  return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):  # dataMatIn-- list 2-dim; classLabels -- list 1-dim  
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                #print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        else:#go over bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                #print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        #print "iteration number: %d" % iter
    return oS.b,oS.alphas # b, alphas --matrix 2-dim indicating b to be value and alphas to be 1-dim, column vector 

def calcWs(alphas,dataArr,classLabels): # alphas -- matrix 2-dim, dataArr-- list 2-dim, classLabels --list 1-dim
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w # w--array 2-dim indicating 1-dim

#ws=calcWs(alphas,dataArr,classLabels),
#dataMat=mat(dataArr)
#s = dataMat[0]*mat(ws) + b
#sign(s[0][0])

def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1)) #svs --mat 2-dim, it is a knowledge base based on support vectors; datMat-- mat 2-dim indicating 1-dim, it is a testing vector,row vector; kernelEval -- mat 2-dim indicating 1-dim column vector
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b #predict --value; labelSV -- mat 2-dim, column vector; alphas[svInd] -- mat 2-dim, column vector
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m)


'''
Adaboost is short for Adaptive Boosting

        feature: numeric,m(benchmark) * n(features)
        labels: set and numeric must be 1 or -1 

external:   
        adaBoostTrainDS() 
        adaClassify()   
'''

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    

def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #init error sum, to +infinity
    for i in range(n):#loop over all dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #calc total error multiplied by D
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40): #dataArr -- list 2-dim, dataSet; classLabels -- list 1-dim
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #init D to all equal
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        #print "D:",D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) 
        D = multiply(D,exp(expon))                             
        D = D/D.sum()
        aggClassEst += alpha*classEst
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        #print "total error: ",errorRate
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst 

def adaClassify(datToClass,classifierArr1):  #datToClass -- list 2-dim or 1-dim, testing vector; classifierArr -- list, adaBoostTrainDS's return value
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    #print "classifierArr", classifierArr1
    for i in range(len(classifierArr1)):
        classEst = stumpClassify(dataMatrix,classifierArr1[i]['dim'],\
                                 classifierArr1[i]['thresh'],\
                                 classifierArr1[i]['ineq'])
        aggClassEst += classifierArr1[i]['alpha']*classEst
        #print aggClassEst
    return sign(aggClassEst)

#classifierArr = adaBoostTrainDS(dataArr, labels, 30)
#adaClassify([0,5,0,5], classifierArr)


'''
extend classlabels(-1,1) or (0,1) algorithm(Lr,svm,adaboosting) to mulitple classlabels

input: dataSet: m(benchmark) * n(features)
       labels: multiple set and numeric 

'''
def one2multi_split(labels,notInLabelsOrOC, OC=(1,-1)): # dataSet --list 2-dim; labels--list 1-dim; OC = (1,-1) or (1,0)
    if notInLabelsOrOC in set(labels).union(set([1,-1,0])):
        print "retry the notInLabelsOrOC"
        sys.exit(-1)
    class_nums = len(list(set(labels)))
    labels_output = {}
    for k in set(labels):
        labelMat = mat(labels).transpose()
        labelMat[nonzero(labelMat.A==k)[0]] = notInLabelsOrOC
        labelMat[nonzero(labelMat.A!=notInLabelsOrOC)[0]] = OC[1]
        labelMat[nonzero(labelMat.A==notInLabelsOrOC)[0]] = OC[0]
        labels_output[k] = list(array(labelMat)[:,0])

    return labels_output

def one2multi_classify(dataSet, labels_dict, testing_vector): # dataSet --list 2-dim; labels_dict -- labels_output; testing_vector --list 1-dim 
    return_class = {} 
    return_value = -1
    for k in labels_dict.keys():

        #adaboosting
        classifierArr = adaBoostTrainDS(dataSet, labels_dict[k], 30)
        return_class[k] = int(adaClassify(testing_vector, classifierArr))
        
        #LR
        trainWeights = stocGradAscent1_lr(concatenate((ones((shape(dataSet)[0],1)), array(dataSet)), axis = 1), labels_dict[k], 1000)
        return_class[k] = int(classifyVector_lr(array(testing_vector), trainWeights))

        #SVM
        b,alphas = smoP(dataSet, labels_dict[k], 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
        datMat=mat(dataSet); labelMat = mat(labels_dict[k]).transpose()
        svInd=nonzero(alphas.A>0)[0]
        sVs=datMat[svInd] #get matrix of only support vectors
        labelSV = labelMat[svInd];
        print "there are %d Support Vectors" % shape(sVs)[0]
        kernelEval = kernelTrans(sVs,mat(testing_vector),('rbf', k1)) #svs --mat 2-dim, it is a knowledge base based on support vectors; datMat-- mat 2-dim indicating 1-dim, it is a testing vector,row vector; kernelEval -- mat 2-dim indicating 1-dim column vector
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b #predict --value; labelSV -- mat 2-dim, column vector; alphas[svInd] -- mat 2-dim, column vector
        return_class[k] = int(sign(predict))

    if return_class.values().count(1) == 1:
        print "only one proper class found, good"
        for k in return_class.keys():
            if return_class[k] == 1:
                return_value = k
        if not return_value == return_class.keys()[return_class.values().index(1)]:
            print "dict index wrong"
        else:
            return return_class.keys()[return_class.values().index(1)]
    elif return_class.values().count(1) > 1: 
        print "more than one proper class found"
        return return_class.keys()[return_class.values().index(1)]
    else:
        print "no proper class found"
        return return_class.keys()[0]
        


def one2one_split(dataSet, labels, notInLabelsOrOC, OC=(1,-1)): # dataSet --list 2-dim; labels--list 1-dim; OC = (1,-1) or (1,0)

    for x in notInLabelsOrOC:
        if x in set(labels).union(set([1,-1,0])):
            print "retry the notInLabelsOrOC"
            sys.exit(-1)

    dataMat = mat(dataSet)
    class_nums = len(list(set(labels)))
    labels_output = {}
    dataSet_output = {}
    labels_unique = list(set(labels))
    for k1 in range(len(labels_unique)):
        for k2 in range(k1+1, len(labels_unique)):


            labelMat = mat(labels).transpose()
            
            set_tmp = set(nonzero(labelMat.A==labels_unique[k1])[0]).union(set(nonzero(labelMat.A==labels_unique[k2])[0]))
            array_tmp = array(list(set(range(len(labels))).difference(set_tmp)))


            if not len(array_tmp) == 0:
                labelMat[array_tmp] = notInLabelsOrOC[0]

            labelMat[nonzero(labelMat.A==labels_unique[k1])[0]] = notInLabelsOrOC[1]

            set_tmp1 = set(nonzero(labelMat.A==notInLabelsOrOC[0])[0]).union(set(nonzero(labelMat.A==notInLabelsOrOC[1])[0]))
            array_tmp1 = array(list(set(range(len(labels))).difference(set_tmp1)))

            #if not len(array_tmp1) == 0:
            labelMat[array_tmp1] = OC[1]
            labelMat[nonzero(labelMat.A==notInLabelsOrOC[1])[0]] = OC[0]


            sub_list_array = sort(concatenate((nonzero(labelMat.A==OC[0])[0],nonzero(labelMat.A==OC[1])[0]), axis = 0))

            #print "sub_list_array", sub_list_array

            dataSet_output[(labels_unique[k1],labels_unique[k2])] = dataMat[sub_list_array]
            labels_output[(labels_unique[k1],labels_unique[k2])] = list(array(labelMat[sub_list_array])[:,0])

    return dataSet_output, labels_output # dataSet_output -- mat 2-dim; labels_output -- list 1-dim


def one2one_classify(dataSet_dict, labels_dict, testing_vector): # dataSet_dict --dataSet_output, dict with mat 2-dim; labels_dict -- labels_output; testing_vector --list 1-dim 
    return_class = {} 
    return_value = -1
    for k1,k2 in labels_dict.keys():

        #adaboosting
        classifierarr,orange = adaboosttrainds(dataset_dict[(k1,k2)], labels_dict[(k1,k2)], 30)
        return_class[(k1,k2)] = int(adaclassify(testing_vector, classifierarr))
        
        #lr
        trainweights = stocgradascent1_lr(concatenate((ones((shape(dataset_dict[(k1,k2)])[0],1)), array(dataset_dict[(k1,k2)])), axis = 1), labels_dict[(k1,k2)], 1000)
        return_class[(k1,k2)] = int(classifyvector_lr(array(testing_vector), trainweights))

        #svm
        b,alphas = smop(dataset_dict[(k1,k2)], labels_dict[(k1,k2)], 200, 0.0001, 10000, ('rbf', k1)) #c=200 important
        datmat=mat(dataset_dict[(k1,k2)]); labelmat = mat(labels_dict[(k1,k2)]).transpose()
        svind=nonzero(alphas.a>0)[0]
        svs=datmat[svind] #get matrix of only support vectors
        labelsv = labelmat[svind];
        print "there are %d support vectors" % shape(svs)[0]
        kerneleval = kerneltrans(svs,mat(testing_vector),('rbf', k1)) #svs --mat 2-dim, it is a knowledge base based on support vectors; datmat-- mat 2-dim indicating 1-dim, it is a testing vector,row vector; kerneleval -- mat 2-dim indicating 1-dim column vector
        predict=kerneleval.t * multiply(labelsv,alphas[svind]) + b #predict --value; labelsv -- mat 2-dim, column vector; alphas[svind] -- mat 2-dim, column vector
        return_class[(k1,k2)] = int(sign(predict))

    return_class_final = {}
    for k1,k2 in labels_dict.keys():
        if return_class[(k1,k2)] == 1:
            return_class_final[k1] = return_class.get(k1,0) + 1
        else:
            return_class_final[k2] = return_class.get(k2,0) + 1

    sortedClassCount = sorted(return_class_final.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]



def datingClassTest3():
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #print "before"
    #print normMat
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)

    normMatPlus = add1forReg(normMat)
    #print "after"
    #print normMatPlus
    w = standRegres(normMatPlus[numTestVecs:m,:],datingLabels[numTestVecs:m])
    print "coefficent"
    print w
    yHat = mat(normMatPlus[0:numTestVecs,:]) * w
    yMat = mat(datingLabels[0:numTestVecs])
    print "error"
    print rssError(yMat.A, yHat.T.A)
    print "corrcoef"
    print corrcoef(yHat.T, yMat)

#datingClassTest3()




def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #print "in autnNorm"
    #print minVals, maxVals 
    ranges = maxVals - minVals
    #print "shape", shape(dataSet)
    normDataSet = zeros(shape(dataSet))
    #print "shape", normDataSet
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    #print "before"
    #print normDataSet
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    #print "after"
    #print normDataSet
    return normDataSet

def add1forReg(xyArr): # xyArr -- array 2-dim or 1-dim, can not use 2-dim indicating 1-dim
    banana = xyArr.copy()
    #print xyArr.shape
    if len(xyArr.shape) == 1:
        outXyArr = concatenate((array([1]),banana), axis=0)
    elif len(xyArr.shape) == 2:
        if xyArr.shape[0] == 1:
            outXyArr = concatenate((array([1]),banana[0]),axis=0)
        else:
            array1Plus = ones((1,xyArr.shape[0]))
            outXyArr = concatenate((array1Plus.T,banana),axis=1)
    else:
        print "input add1forReg is wrong"
        sys.exit(-1)
    return outXyArr

'''
regression is including:
    standregression, local weight linear regression and ridge regression

        feature: numeric,m(benchmark) * n(features)
        labels: numeric

'''
def standregres(xarr,yarr): #xarr -- array/list 2-dim, yarr -- array/list 1-dim
    xmat = mat(xarr); ymat = mat(yarr).t
    xtx = xmat.t*xmat
    if linalg.det(xtx) == 0.0:
        print "this matrix is singular, cannot do inverse"
        return
    ws = xtx.i * (xmat.t*ymat)
    return ws #matrix 2-dim indicating 1-dim column vector

def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()


def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr) 
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat


def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = denom.I * (xMat.T*yMat)
    return ws
    
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #to eliminate X0 take mean off of Y
    #regularize X's
    xMeans = mean(xMat,0)   #calc mean then subtract it off
    xVar = var(xMat,0)      #calc variance of Xi then divide by it
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

if __name__ == "__main__":
    infile=open(sys.argv[1],'r')
    benchmark={}
    metric=[]
    benchmark_time={}
    line=infile.readline().strip()
    benchmark_name=[]
    while(line):
        smt_time={}
        benchmark_name.append(line)
        for i in xrange(8):
            line=infile.readline().strip().split(':')
            smt=line[0]
            run_time=line[-1]
            #print smt
            #print "run_time", run_time
            if benchmark_time.has_key(smt):
                benchmark_time[smt].append(float(run_time))
            else:
                tmptime=[float(run_time)]
                benchmark_time[smt]=tmptime
            data=line[1].split(',')[:]
            metric=[float(j) for j in data]
            if benchmark.has_key(smt):
                benchmark[smt].append(metric)
            else:
                newlist=[metric]
                benchmark[smt]=newlist
        line=infile.readline().strip()
    infile.close()
    time_file=open(sys.argv[2],'r')
    line=time_file.readline().strip()
    bench={}
    #name=''
    while(line):
        smt_tmp={}
        name=line
        for i in xrange(8):
            line=time_file.readline().strip().split(':')
            smt=line[0]
            run_time=line[-1]
            #print smt
            #print "run_time", run_time
            smt_tmp[smt]=float(run_time)
        bench[name]=smt_tmp
        line=time_file.readline().strip()
    time_file.close()
 
    #print benchmark['1']
    #print len(benchmark_time['2'])
    #print benchmark_time['2']
    
    #spark-sort
    #1:17597389120,17914074929,84345695799,5602550799,11023516509,36054331456,376987960,519241337,725460525,5300247208594:93.6982560158
    #2:32195823247,23999249017,95727276985,7693899129,12830732006,39327650334,1182518579,1167128830,775561310,7703878575537:78.1003930569
    #4:43423900393,33199997075,113271992543,10724305905,16527968433,42520319696,2955812426,2076257282,995526782,11523007294823:74.7541120052
    #8:68330129332,49701861943,143206315233,14752932032,21095885191,50040318018,6794160477,4486392371,1960141642,23589425534495:97.6858568192
    
#starting: regression
    #test base
    #hoRatio = 0.10      #hold out 10%
    #normMat = autoNorm(array(benchmark['4']))
    #m = normMat.shape[0]
    #numTestVecs = int(m*hoRatio)
    #normMatPlus = add1forReg(normMat)
    #datingLabels = list(array(benchmark_time['2'])/array(benchmark_time['4']))

    ## core predict
    #yHat = zeros(numTestVecs)
    #for i in range(numTestVecs):
    #    yHat[i] = lwlr(normMatPlus[i,:], normMatPlus[numTestVecs:m,:],datingLabels[numTestVecs:m])
#    test =[ [17597389120.0,17914074929.0,84345695799.0,5602550799.0,11023516509.0,36054331456.0,376987960.0,519241337.0,725460525.0,5300247208594.0],[32195823247,23999249017,95727276985,7693899129,12830732006,39327650334,1182518579,1167128830,775561310,7703878575537],[43423900393,33199997075,113271992543,10724305905,16527968433,42520319696,2955812426,2076257282,995526782,11523007294823],[68330129332,49701861943,143206315233,14752932032,21095885191,50040318018,6794160477,4486392371,1960141642,23589425534495]]
#    test_time=[93.7,78.1,74.75,97,7]
#    #print len(test_one)
#    #print len(benchmark['1'][0])
#    #testMat = autoNorm(array(test_one))
#    #print "testMat",testMat
#    #testPlus = add1forReg(testMat)
#    #print "tsetPlus:",testPlus
#    #print type(benchmark['1'])
#    
#    def predict(test, test_time, smt_level):
#        Test_mat=benchmark[smt_level]
#        Test_mat.append(list(array(test[:-1])/test_time))
#        matA=array(Test_mat)
#        normMat = autoNorm(matA)
#        normMatPlus = add1forReg(normMat)
#        for i in xrange(4):
#           #print "shape:",shape(normMatPlus),normMatPlus
#            #print 2**i
#            #print shape(benchmark_time[str(2**i)])
#            datingLabels = list(array(benchmark_time[str(2**i)])/array(benchmark_time[smt_level]))
#            yHat = lwlr(normMatPlus[-1,:], normMatPlus[:-1,:],datingLabels)
#            print "yHat is",yHat
#    for i in xrange(4):
#        print "smt is",2**i
#        predict(test[i],test_time[i],str(2**i))

#starting: by, knn
#    for smtI in ["1","2","3","4","5","6","7","8"]:
#        hoRatio = 0.2 #hold out 10%
#        datingDataMat = array(benchmark[smtI])
#        datingLabels  = benchmark_time[smtI]
#        normMat = autoNorm(datingDataMat)
#        #print normMat
#        m = normMat.shape[0]
#        #numTestVecs = int(m*hoRatio)
#        numTestVecs = 11 #int(m*hoRatio)
#
#        outData = quantize(normMat[numTestVecs:m],[0.0001,0.0001,0.0001,0.00002,0.00002,0.0005,0.00001,0.00001,0.00001,0.1]) 
#        outData1 = zeros((shape(outData)[0], shape(outData)[1]), dtype = int)
#        for i in range(shape(outData)[0]):
#            for j in range(shape(outData)[1]):
#                outData1[i,j] = int(outData[i,j]*100000)
#
#        px_y, py = build_bayes(outData1, datingLabels[numTestVecs:m], lamda=1.0)  #dataSet--numpy array:2-dim,discrete; labels--list,discrete; lamda--laplace smoothing, by
#
#        errorCount = 0.0
#        for i in range(numTestVecs):
#            #classifierResult = classify_knn(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3) #knn,1:test
#            outX = quantize_match(outData1, normMat[i,:]*100000)    #inData--numpy array: 2-dim, inX--list #by
#            classifierResult = classify_bayes(outX, px_y, py) #by
#
#            if (classifierResult != datingLabels[i]): 
#                actR = str(int(classifierResult))
#                expect = str(int(datingLabels[i]))
#                dif=abs(bench[name][expect]-bench[name][actR])/bench[name][expect]
#                errorCount += dif
#        print "SMT is",smtI,"the total error rate is: %f" % (errorCount/float(numTestVecs))
#        print errorCount
#            classifierResult = str(int(sortedClassCount[0][0]))
#            espectResult = str(int(datingLabels[i]))
#
#            if (classifierResult != espectResult):
#                #print "result is", type(classifierResult)
#                name=benchmark_name[i]
#                dif=abs(bench[name][espectResult]-bench[name][classifierResult])/bench[name][espectResult]
#                errorCount += dif #1.0
#
#        print "SMT is", smtI ,"the total error rate is: %f" % (errorCount/float(numTestVecs))
#

#starting: lr,bs,svm
    for smtI in ["1","2","3","4","5","6","7","8"]:
        hoRatio = 0.3 #hold out 10%
        datingDataMat = array(benchmark[smtI])
        datingLabels  = benchmark_time[smtI]
        #print datingDataMat.shape
        #print len(datingLabels)

        #datingDataMat,datingLabels = file2matrix('datingTestSet2.txt',3)       #load data setfrom file
        #datingDataMat,datingLabels = file2matrix('testSet.txt',2)       #load data setfrom file

        normMat = autoNorm(datingDataMat)
        #normMat = datingDataMat
        m = normMat.shape[0]
        #numTestVecs = int(m*hoRatio)
        numTestVecs = 11#int(m*hoRatio)


        datingMat_output, datingLabels_output = one2one_split(normMat[numTestVecs:m,:], datingLabels[numTestVecs:m],(100,101),(1,0)) #LR
        
        #datingMat_output, datingLabels_output = one2one_split(normMat[numTestVecs:m,:], datingLabels[numTestVecs:m],(100,101)) #SVM, BOOSTING

        #print "after split"
        #print datingLabels_output

        classifierArr = {}
        trainWeights = {}
        kerneleval = {}
        b = {}
        alphas = {}
        svs = {}
        svind = {}
        labelsv = {}
        for k1,k2 in datingLabels_output.keys():

            #print k1,k2
            #adaboosting
            #classifierArr[(k1,k2)],orange = adaBoostTrainDS(datingMat_output[(k1,k2)], datingLabels_output[(k1,k2)], 30)

            #lr
            trainWeights[(k1,k2)] = stocGradAscent1_lr(add1forReg(datingMat_output[(k1,k2)].A), datingLabels_output[(k1,k2)], 1000)


            #svm
            #b[(k1,k2)],alphas[(k1,k2)] = smoP(datingMat_output[(k1,k2)], datingLabels_output[(k1,k2)], 5, 0.0001, 10000, ('lin', 0)) #c=200 important
            #datmat=mat(datingMat_output[(k1,k2)]); labelmat = mat(datingLabels_output[(k1,k2)]).transpose()
            #svind[(k1,k2)]=nonzero(alphas[(k1,k2)].A>0)[0]
            #svs[(k1,k2)]=datmat[svind[(k1,k2)]] #get matrix of only support vectors
            #labelsv[(k1,k2)] = labelmat[svind[(k1,k2)]];
            ##print "there are %d support vectors" % shape(svs[(k1,k2)])[0]
        filename = "./LR" + "_" + smtI + ".txt"
        fw = open(filename,'w')
        pickle.dump(trainWeights,fw)
        print "train in : ", trainWeights
        fw.close()
#

        errorCount = 0.0

        fr = open("./LR_" + str(smtI) +".txt")
        TrainW = pickle.load(fr)
        fr.close()
        print "train out : ", TrainW 
        for i in range(numTestVecs):
            return_class = {} 
            kerneleval = {}
            for k1,k2 in datingLabels_output.keys():
                #return_class[(k1,k2)] = int(adaClassify(normMat[i,:], classifierArr[(k1,k2)])) #bs
                #return_class[(k1,k2)] = int(classifyVector_lr(add1forReg(array(normMat[i,:])), trainWeights[(k1,k2)]))#lr
                return_class[(k1,k2)] = int(classifyVector_lr(add1forReg(array(normMat[i,:])), TrainW[(k1,k2)]))#lr
                #kerneleval[(k1,k2)] = kernelTrans(svs[(k1,k2)],mat(normMat[i,:]),('lin', 0)) #svs --mat 2-dim, it is a knowledge base based on support vectors; datmat-- mat 2-dim indicating 1-dim, it is a testing vector,row vector; kerneleval -- mat 2-dim indicating 1-dim column vector
                #predict=kerneleval[(k1,k2)].T * multiply(labelsv[(k1,k2)],alphas[(k1,k2)][svind[(k1,k2)]]) + b[(k1,k2)] #predict --value; labelsv -- mat 2-dim, column vector; alphas[svind] -- mat 2-dim, column vector
                ###print predict
                #return_class[(k1,k2)] = int(sign(predict))#svm

            return_class_final = {}
            for k1,k2 in datingLabels_output.keys():
                if return_class[(k1,k2)] == 1:
                    return_class_final[k1] = return_class_final.get(k1,0) + 1
                else:
                    return_class_final[k2] = return_class_final.get(k2,0) + 1

            #print return_class
            #print return_class_final

            sortedClassCount = sorted(return_class_final.iteritems(), key=operator.itemgetter(1), reverse=True)
            classifierResult = str(int(sortedClassCount[0][0]))
            espectResult = str(int(datingLabels[i]))

            if (classifierResult != espectResult):
                #print "result is", type(classifierResult)
                name=benchmark_name[i]
                dif=abs(bench[name][espectResult]-bench[name][classifierResult])/bench[name][espectResult]
                errorCount += dif #1.0

        print "SMT is", smtI ,"the total error rate is: %f" % (errorCount/float(numTestVecs))
        #print errorCount
def read_base(base):
    infile=open(base,'r')
    benchmark={}
    metric=[]
    benchmark_time={}
    line=infile.readline().strip()
    benchmark_name=[]
    while(line):
        smt_time={}
        benchmark_name.append(line)
        for i in xrange(8):
            line=infile.readline().strip().split(':')
            smt=line[0]
            run_time=line[-1]
            #print smt
            #print "run_time", run_time
            if benchmark_time.has_key(smt):
                benchmark_time[smt].append(float(run_time))
            else:
                tmptime=[float(run_time)]
                benchmark_time[smt]=tmptime
            data=line[1].split(',')[:]
            metric=[float(j) for j in data]
            if benchmark.has_key(smt):
                benchmark[smt].append(metric)
            else:
                newlist=[metric]
                benchmark[smt]=newlist
        line=infile.readline().strip()
    infile.close()
    return benchmark, benchmark_time

def read_perfoutput_period(name):
    #global inter_lines
    event_norm_map=[-1,-1,0,0,0,0,0,0,0,0,0,0]
    event_nums = len(event_norm_map)
    counters=np.array([0]*event_nums)
    #sample=[0]*(event_nums-2)
    f1 = open(name, "r")
    #f1 = open(name, "r")
    line_index = 0
    event_id = 0
    #flag = 0
    #last_line = ''
    while True:
    	line = f1.readline().strip()
        line_index += 1
        #print line
        if line_index <= 3: #+ time_idx*event_nums + inter_lines:       continue
            continue
    	if len(line) == 0:
            break
            #flag = -1
            #print "perf result is late"
    	if line.split()[0].find(".") == -1:
            #inter_lines += 1
            continue
    	if line.split()[1].find("not") == -1:
            counters[event_id] += int(line.split()[1])
        else:
            counters[event_id] += 0 
        event_id += 1
        #last_line=line
        if event_id >= event_nums:
            event_id=0
            #print "perf result is ok"
            #break
    f1.close()
    #print "time %d counters" % time_idx
    #print counters
    #for i in range(2,len(counters)):
    #    sample[i-2] = float(counters[i])/counters[event_norm_map[i]]
    #counters=counters/float(counters[0])
    #sample[1:(event_nums-1)]=counters[2:event_nums]
    #return counters #,flag
    return counters


def LR_predict(base_file,perf_file,smt):
    if (os.path.getsize(perf_file)==0) :
        return 0
    test = read_perfoutput_period(perf_file)
    counters = array([0.0]*(len(test)-2))
    for i in range(2,len(test)):
        counters[i-2] = float(test[i])/test[0]
    #counter
    benchmark,benchmark_time = read_base(base_file)
    datingLabels  = benchmark_time[smt]
    base_list=benchmark[smt]
    base_list.append(list(counters))
    datingDataMat = array(base_list)
    normMat =autoNorm(datingDataMat)
    fr = open("/home/PBDST/models/LR_" + str(smt) +".txt")
    TrainW = pickle.load(fr)
    fr.close()
    #print TrainW
    #print "train out : ", TrainW 
    return_class = {} 
    kerneleval = {}
    datingMat_output, datingLabels_output = one2one_split(normMat[:-1,:], datingLabels,(100,101),(1,0)) #LR
    
    for k1,k2 in datingLabels_output.keys():
        #print TrainW[(k1,k2)]
        return_class[(k1,k2)] = int(classifyVector_lr(add1forReg(array(normMat[-1,:])), TrainW[(k1,k2)]))#lr
    return_class_final = {}
    for k1,k2 in datingLabels_output.keys():
        if return_class[(k1,k2)] == 1:
            return_class_final[k1] = return_class_final.get(k1,0) + 1
        else:
            return_class_final[k2] = return_class_final.get(k2,0) + 1
    sortedClassCount = sorted(return_class_final.iteritems(), key=operator.itemgetter(1), reverse=True)
    return int(sortedClassCount[0][0])

def KNN_predict(base_file,perf_file,smt):
    if (os.path.getsize(perf_file)==0) :
        return 0
    test = read_perfoutput_period(perf_file)
    counters = array([0.0]*(len(test)-2))
    for i in range(2,len(test)):
        counters[i-2] = float(test[i])/test[0]
    #counter
    benchmark,benchmark_time = read_base(base_file)
    datingLabels  = benchmark_time[smt]
    base_list=benchmark[smt]
    base_list.append(list(counters))
    datingDataMat = array(base_list)
    normMat =autoNorm(datingDataMat)
    classifierResult = classify_knn(normMat[-1,:],normMat[:-1,:],datingLabels,1) #knn,1:test
    #print classifierResult 
    return int(classifierResult)
#starting: tree
##labels: set
#    for smtI in ["1","2","3","4","5","6","7","8"]:
#        TOLS=0.01
#        TOLN=1
#        numTestVecs = 11 #int(m*hoRatio)
#        #outData = quantize(normMat[numTestVecs:m],[0.0001,0.0001,0.0001,0.00002,0.00002,0.0005,0.00001,0.00001,0.00001,0.1]) 
#        dataSet=[]
#        for i in xrange(len(benchmark[smtI][numTestVecs:])):
#            #print benchmark[smtI][numTestVecs+i],benchmark_time[smtI][numTestVecs+i]
#            dataSet.append(benchmark[smtI][numTestVecs+i]+[benchmark_time[smtI][numTestVecs+i]])
#        myMat = mat(dataSet)
#        myTree = createTree(myMat, GiniLeaf, Gini, (TOLS,TOLN))
#        filename = "./tree" + "_" + smtI + ".txt"
#        fw = open(filename,'w')
#        pickle.dump(myTree,fw) 
#        fw.close()
#    
#    for smtI in ["1","2","3","4","5","6","7","8"]:
#        errorCount = 0.0
#        for i in xrange(numTestVecs):
#            sample=benchmark[smtI][i]
#            feature_vector = mat(sample)
#            fr = open("./tree_" + str(smtI) +".txt")
#            myTree = pickle.load(fr)
#            conf_vector = createForeCast(myTree, feature_vector, GiniTreeEval)[0]
#            fr.close()
#            actR = str(conf_vector[0])
#            expectR = str(int(benchmark_time[smtI][i]))
#            if (actR != expectR):
#                name=benchmark_name[i]
#                dif=abs(bench[name][expectR]-bench[name][actR])/bench[name][expectR]
#                errorCount += dif #1.0
#        print "SMT is", smtI ,"the total error rate is: %f" % (errorCount/float(numTestVecs))
##
