import numpy as np
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import WeightRegularizer
from keras.wrappers.scikit_learn import KerasClassifier
import os
import pickle

methodStrPrefix = ''
outputFolder = './TrainResults-Train20Py/'
outputFolderForFolds = outputFolder + 'OutputForFolds/'

def readInputFile(inputFile, numInputs, enhancerNamesColumn=0, classColumn=1, firstInputColumn=5):
  
  # First row has feature input names, remaining rows have enhancer names and feature values starting at firstInputColumn
  inputNames = open(inputFile, 'r').readline().split('\t')[firstInputColumn:firstInputColumn+numInputs]  
  data = np.genfromtxt(inputFile, skip_header=1, usecols=range(firstInputColumn, firstInputColumn + numInputs))

  # Read class labels of inputs and enhancer names using the given column numbers
  classLabels = np.genfromtxt(inputFile, names=True, usecols=(classColumn), dtype=None, 
            converters={"Class": lambda x: 1 if x == b'positive' or x == b'1' else 0})["Class"]
  enhancerNames = np.genfromtxt(inputFile, names=True, usecols=(enhancerNamesColumn), dtype=None, encoding=None)
  
  # Calculate the Z-scores of the feature inputs.  From:
  # https://datascience.stackexchange.com/questions/13178/how-to-normalize-data-for-neural-network-and-decision-forest
  #
  # Scaling data: not needed for random forests.
  # For NN training: necessary for weight regularization, helpful for regular SGD variants of backpropagation.
  # Recommends linear scaling around the mean in the range -0.5 to +0.5 or using standard score 
  # (# standard deviations from the mean, see https://en.wikipedia.org/wiki/Standard_score)
  # 
  # StandardScaler in sklearn normalizes with standard score by default.
  #
  # From http://scikit-learn.org/stable/modules/svm.html:
  #
  # Support Vector Machine algorithms are not scale invariant, so it is highly recommended to scale your data. 
  # For example, scale each attribute on the input vector X to [0,1] or [-1,+1], or standardize it to have mean 0 
  # and variance 1. Note that the same scaling must be applied to the test vector to obtain meaningful results.        
  #
  scaler = StandardScaler(with_mean=True, with_std=True)
  scaler.fit(data)
  scaledData = scaler.transform(data)
  del data
  del scaler
  return (enhancerNames, inputNames, scaledData, classLabels)

def buildNN(optimizer='rmsprop', l2=1e-2, dropout=0.25, activation='relu', units=(20, 20), inputShape=0):
  model = Sequential()

  # units is a list with hte number of hidden units in each layer. Must proide input_shapefor first neural net layer.
  model.add(Dense(units[0], W_regularizer = WeightRegularizer(l1=0, l2=l2), activation=activation, input_shape=inputShape))
  model.add(Dropout(dropout))
  for j in range(1, len(units)):
    model.add(Dense(units[j], W_regularizer = WeightRegularizer(l1=0, l2=l2), activation=activation))
    model.add(Dropout(dropout))

  # Add a layer iwth 1 unit and sigmoid activation
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return model

def train(runNumber, fold, 
          classifierTypes,                  # List of classifier/data balance combinations to train
          dataFoldTrainImbalanced,          # Training data for the fold
          classLabelsFoldTrainImbalanced,   # Class labels for the training data
          dataFoldVal,                      # Validation data for the fold
          classLabelsFoldVal,               # Class labels for the validation data
          randomState, nnArgsDict):

  np.random.seed(randomState + 2000)
  random.seed(randomState + 2000)

  positiveIndices = (classLabelsFoldTrainImbalanced == 1).nonzero()[0]
  negativeIndices = (classLabelsFoldTrainImbalanced == 0).nonzero()[0]
  imbalanceRatio = len(negativeIndices) // len(positiveIndices)

  dataFoldIndicesTrainUndersample = []
  for ctr in range(imbalanceRatio + 2):
    dataFoldIndicesTrainUndersample.append(np.concatenate((positiveIndices,
      random.sample(list(negativeIndices), len(positiveIndices)))))
  classLabelsFoldTrainUndersample = [classLabelsFoldTrainImbalanced[x] for x in dataFoldIndicesTrainUndersample]
  dataFoldTrainUndersample = [dataFoldTrainImbalanced[x] for x in dataFoldIndicesTrainUndersample]
  
  dataFoldIndicesTrainOversample = np.concatenate((positiveIndices,
    np.random.choice(positiveIndices, len(negativeIndices) - len(positiveIndices)), # With replacement
    negativeIndices))
  classLabelsFoldTrainOversample = classLabelsFoldTrainImbalanced[dataFoldIndicesTrainOversample]
  dataFoldTrainOversample = dataFoldTrainImbalanced[dataFoldIndicesTrainOversample]
  
  print("Run Number:", runNumber, " Fold:", fold)
  print('Samples: Train Unbalanced:  ', len(classLabelsFoldTrainImbalanced), '=', len(negativeIndices), '+', len(positiveIndices), '; Imbalance Ratio:', imbalanceRatio)
  print('Samples: Train Oversample:  ', len(classLabelsFoldTrainOversample), '=', list(classLabelsFoldTrainOversample).count(0), '+', list(classLabelsFoldTrainOversample).count(1))
  print('Samples: Train Undersample: ', len(classLabelsFoldTrainUndersample[0]), '=', list(classLabelsFoldTrainUndersample[0]).count(0), '+', list(classLabelsFoldTrainUndersample[0]).count(1))
  print('Samples: Train UndersampleMany uses', len(dataFoldTrainUndersample), 'models in ensemble for each fold')
  print('Val: ', len(classLabelsFoldVal), '= (', list(classLabelsFoldVal).count(0), '+', list(classLabelsFoldVal).count(1))

  for classifierType in classifierTypes:
    # Initialize random seeds in both np.random and the Python random library in case any of the classifiers
    # uses them to generate random numbers
    np.random.seed(randomState + 1000)
    random.seed(randomState + 1000)
    outputFileFoldIterPrefix = outputFolderForFolds + classifierType + "-Run" + str(runNumber) + "-Fold" + str(fold) + "-"
    ensembleModels = 1
    if 'OS' in classifierType:
      dataFoldTrain = dataFoldTrainOversample
      classLabelsFoldTrain = classLabelsFoldTrainOversample
    elif 'USM' in classifierType: 
      ensembleModels = len(dataFoldTrainUndersample)
    else:
      dataFoldTrain = dataFoldTrainImbalanced
      classLabelsFoldTrain = classLabelsFoldTrainImbalanced

    for ctr in range(ensembleModels):
      print ('    ', classifierType, ctr)

      outputFileFoldIterPrefixForClassifier = outputFileFoldIterPrefix
      if 'US' in classifierType:
        dataFoldTrain = dataFoldTrainUndersample[ctr]
        classLabelsFoldTrain = classLabelsFoldTrainUndersample[ctr]
        if 'USM' in classifierType: 
          outputFileFoldIterPrefixForClassifier = outputFileFoldIterPrefix + 'USM' + str(ctr) + '-'

      nnArgs = None
      if 'NN' in classifierType:
        for key in nnArgsDict.keys():
          if key in classifierType: nnArgs = nnArgsDict[key]          
        nnArgs['inputShape'] = (dataFoldTrain.shape[1],)
        nnArgs['validation_data'] = (dataFoldVal, classLabelsFoldVal)
        if 'CW' in classifierType:
          wtDict = {}
          wtDict[0] = len(classLabelsFoldTrain)/(2.0 * np.bincount(classLabelsFoldTrain)[0])
          wtDict[1] = len(classLabelsFoldTrain)/(2.0 * np.bincount(classLabelsFoldTrain)[1])
          nnArgs['class_weight'] = wtDict
        classifier = KerasClassifier(build_fn=buildNN, **nnArgs)
      elif 'RF' in classifierType:
        classifier = RandomForestClassifier(random_state=np.random.RandomState(randomState+1000),
                       n_estimators = 1000, class_weight = "balanced" if 'CW' in classifierType else None)
      elif 'SVMLin' in classifierType:
        # Same as LinearSVC but supports probability for predictions and enables predict_proba()
        # LinearSVC does not support that
        classifier = SVC(kernel='linear', 
                     random_state=np.random.RandomState(randomState+1000),
                     class_weight = "balanced" if 'CW' in classifierType else None)
      elif 'SVMRad' in classifierType:
        # Defaults to Radial Basis Function (kernel='rbf') with C=1
        classifier = SVC(kernel = 'rbf',   # Radial Basis Function; this is the default value for this parameter
                     random_state=np.random.RandomState(randomState+1000),
                     class_weight = "balanced" if 'CW' in classifierType else None) 
      elif 'LogisticReg' in classifierType:
        # Use penalty = 'l1' for L1 regularization. Only 'liblinear’ solver is supported for penalty = 'l1'
        # defaults are penalty = 'l2' and solver = 'liblinear'
        classifier = linear_model.LogisticRegression(penalty = 'l2' if 'L2' in classifierType else 'l1', 
                     solver = 'liblinear',
                     random_state=np.random.RandomState(randomState+1000),
                     class_weight = "balanced" if 'CW' in classifierType else None) 
      else:
        classifier = None

      # Train the model
      classifier.fit(dataFoldTrain, classLabelsFoldTrain)

      # Save the trained model
      if 'NN' in classifierType:
        classifier.model.save(outputFileFoldIterPrefixForClassifier + "Classifier.h5")
      else:
        clfFile = open(outputFileFoldIterPrefixForClassifier + "Classifier.pkl", 'wb')
        pickle.dump(classifier, clfFile)
        clfFile.close()

      #  Free memory
      del classifier 
      if 'NN' in classifierType:
        del nnArgs['inputShape']
        del nnArgs['validation_data']
        nnArgs.pop('class_weight', None) # Remove key if it exists
        nnArgs.pop('callbacks', None) 

      ######
      # End of for ensembleModels loop
      ######

    #####
    # End of for classifier loop
    #####

  # Free memory
  del positiveIndices, negativeIndices, 
  del dataFoldIndicesTrainUndersample, classLabelsFoldTrainUndersample, dataFoldTrainUndersample
  del dataFoldIndicesTrainOversample, classLabelsFoldTrainOversample, dataFoldTrainOversample

def run(allClassifierTypes, runNumber, numFolds):

  if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

  if not os.path.exists(outputFolderForFolds):
    os.makedirs(outputFolderForFolds)

  enhancerNames, inputNames, data, classLabels = readInputFile("ChromatinRNASeq.tsv", numInputs=46+4)

  # Additional parameters for the Keras neurla network classifier.
  nnArgs2020Relu = dict(l2=1e-2, dropout=0.35, nb_epoch=100, batch_size=32, shuffle=True, units=(20,20), verbose=0)
  nnArgs302010Relu = dict(l2=5e-3, dropout=0.4, nb_epoch=100, batch_size=32, shuffle=True, units=(30,20,10), verbose=0)
  

  nnArgsDict = {'NN-20-20-Relu' : nnArgs2020Relu, 
                'NN-30-20-10-Relu' : nnArgs302010Relu}

  # Test data imbalance is the split in same ratio as the original data. 
  # Note that runNumber is used to calculate the seed for random number generator
  randomState = 1729*(runNumber+1)
  (dataTrain, dataTest, classLabelsTrain, classLabelsTest) = train_test_split(data, 
    classLabels, stratify = classLabels, 
    test_size = 0.2, random_state = np.random.RandomState(randomState), shuffle = True)
  
  # Stratified K-Fold object to generate cross-validation folds
  cv = StratifiedKFold(
    n_splits=numFolds, 
    shuffle=True, 
    random_state=np.random.RandomState(randomState-1000))

  # Train model for each classifier/data balance combination for each fold
  for fold in range(numFolds):
    dataFoldIndicesTrainImbalanced, dataFoldIndicesVal = list(cv.split(dataTrain, classLabelsTrain))[fold]
    dataFoldVal = dataTrain[dataFoldIndicesVal]
    classLabelsFoldVal = classLabelsTrain[dataFoldIndicesVal]
    classLabelsFoldTrainImbalanced = classLabelsTrain[dataFoldIndicesTrainImbalanced]
    dataFoldTrainImbalanced = dataTrain[dataFoldIndicesTrainImbalanced]
    train(runNumber = runNumber, fold = fold, classifierTypes = allClassifierTypes,
          dataFoldTrainImbalanced = dataFoldTrainImbalanced, 
          classLabelsFoldTrainImbalanced = classLabelsFoldTrainImbalanced, 
          dataFoldVal = dataFoldVal,
          classLabelsFoldVal = classLabelsFoldVal,
          randomState = randomState, nnArgsDict = nnArgsDict)
    del dataFoldTrainImbalanced, classLabelsFoldTrainImbalanced, dataFoldVal, classLabelsFoldVal
    del dataFoldIndicesTrainImbalanced, dataFoldIndicesVal


  del dataTrain, dataTest, classLabelsTrain, classLabelsTest


if __name__ == "__main__":

  runNumber = 0  # Change for different runs. Each run splits the data intro training and test differently
  numFolds = 10
  
  classifierTypes = ['RF', 'L1LogisticReg', 'L2LogisticReg', 'SVMLin', 'SVMRad', 'NN-20-20-Relu', 'NN-30-20-10-Relu']

  allClassifierTypes = []
  methodStr = methodStrPrefix + 'SupLng' # Used for supervised learning classifiers
  for c in classifierTypes:
    c = c + '-' + methodStr    
    allClassifierTypes.append(c)  # No data balance method in the classifier name means use imbalanced data
    allClassifierTypes.append(c + '-CW')  # Class weight = miscalculation cost for minority class, algorithmic method
    allClassifierTypes.append(c + '-OS')  # Oversample minority class, data method
    allClassifierTypes.append(c + '-US')  # Undersample majority class, data method
    allClassifierTypes.append(c + '-USM') # UNdersample Ensemble

  run(allClassifierTypes, runNumber, numFolds)
