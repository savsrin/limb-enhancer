import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score
import csv
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

def getMetrics(labels, scoresPositive, scoresAreFromDecisionFunction):
  cutoff = 0 if scoresAreFromDecisionFunction else 0.5
  pred = [1 if x > cutoff else 0 for x in scoresPositive]
  tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
  precision, recall, thresholds = precision_recall_curve(labels, scoresPositive)
  auprc = auc(recall, precision)
  return  (
    ((tp + tn)/(tp + tn + fp + fn),  # Accuracy
    1e-8 if tp + fn == 0 else tp/(tp + fn), # Sensitivity or Recall or TPR
    1e-8 if tn + fp == 0 else tn/(tn + fp), # Specificity or TNR
    1e-8 if tp + fp == 0 else tp/(tp + fp), # Precision or PPV
    1e-8 if tn + fn == 0 else tn/(tn + fn), # NPV
    2*tp/(2*tp + fp + fn), # F1
    roc_auc_score(labels, scoresPositive),
    auprc))

def score(runNumber, fold, 
          classifierType,           # Classifier/data balance combinations to evaluate
          data,                     # Data samples to be scored
          ensembleModels            # Number of models in the ensemble for undersampling ensemble method
        ):

  outputFileFoldIterPrefix = outputFolderForFolds + classifierType + "-Run" + str(runNumber) + "-Fold" + str(fold) + "-"
  numModels = ensembleModels if 'USM' in classifierType else 1
  
  scoresList = []
  for ctr in range(numModels):

    outputFileFoldIterPrefixForClassifier = outputFileFoldIterPrefix
    if 'USM' in classifierType: 
      outputFileFoldIterPrefixForClassifier = outputFileFoldIterPrefix + 'USM' + str(ctr) + '-'
  
    if 'NN' in classifierType:
      model = load_model(outputFileFoldIterPrefixForClassifier + "Classifier.h5")
    else:
      clfFile = open(outputFileFoldIterPrefixForClassifier + "Classifier.pkl", 'rb')
      classifier = pickle.load(clfFile)
      clfFile.close()
      del clfFile

    # Save only the scores of the positive class - that's all that is needed
    if 'SVM' in classifierType:
      scoresList.append(classifier.decision_function(data))
      del classifier
    elif 'NN'  in classifierType:
      predictions = model.predict(data, batch_size = 32)
      positiveScores = []
      for x in predictions:
        positiveScores.append(x[0])
      scoresList.append(positiveScores)
      del predictions, positiveScores, model
    else:
      scoresList.append(classifier.predict_proba(data)[:,1])
      del classifier

   # Average over the ensemble models in the fold. Only the USM data balance method has more than one 
  # ensemble model in the fold
  scores = np.mean(scoresList, axis=0)
  del scoresList
  return scores


def run(allClassifierTypes, runNumber, numFolds):

  enhancerNames, inputNames, data, classLabels = readInputFile("ChromatinRNASeq.tsv", numInputs=46+4)

  # Test data imbalance is the split in same ratio as the original data. 
  # Note that runNumber is used to calculate the seed for random number generator
  randomState = 1729*(runNumber+1)
  (dataTrain, dataTest, classLabelsTrain, classLabelsTest) = train_test_split(data, 
    classLabels, stratify = classLabels, 
    test_size = 0.2, random_state = np.random.RandomState(randomState), shuffle = True)
  
  with open(outputFolder + 'Scores-Summary-Run' + str(runNumber) + '.tsv','w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    # First, write the summary of Mean of Test data for each classifier
    w.writerow(['Summary of results for run number  ' + str(runNumber)])
    w.writerow([''])
    w.writerow(['Test data', '', '', '', '', '', '', '', '', '', 'All data'])
    w.writerow(['Classifier', 'Acc', 'Sn', 'Sp', 'Precision/PPV', 'NPV', 'F1', 'AUROC', 'AUPRC', '',
                              'Acc', 'Sn', 'Sp', 'Precision/PPV', 'NPV', 'F1', 'AUROC', 'AUPRC'])
    for classifierType in allClassifierTypes:
      scoresTestList = []
      scoresDataList = []
      for fold in range(numFolds):
        print("Run Number:", runNumber, " Classifier:", classifierType, " Fold:", fold)
        scoresTestList.append(score(runNumber = runNumber, fold = fold, classifierType = classifierType,
                                data = dataTest, ensembleModels = 10))
        scoresDataList.append(score(runNumber = runNumber, fold = fold, classifierType = classifierType,
                                data = data, ensembleModels = 10))

      scoresTest = np.mean(scoresTestList, axis=0)
      scoresData = np.mean(scoresDataList, axis=0)
      del scoresTestList, scoresDataList
      w.writerow([classifierType] +
                   list(getMetrics(classLabelsTest, scoresTest, 'SVM' in classifierType)) +
                   list(getMetrics(classLabels, scoresData, 'SVM' in classifierType)))


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
