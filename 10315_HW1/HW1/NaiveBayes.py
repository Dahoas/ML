"""
NaiveBayes.py

This is starter code for implementation problem 1.3 in HW01.
It is meant to help you organize your work flow. Refer to the
handout for more details on how to complete this code.

Released: 06/09/2019
Author:   Aliaa Essameldin
"""

########################### Libraries Used #####################################
#                                                                              #
#  Feel free to add more libraries to help you calculate log or pre-process    #
#  input. Please make sure not use any libraries that do the probability       #
#  estimations for you. You have to implement all computations yourself.       #
#                                                                              #
################################################################################

import csv
import numpy as np


####################### Pre-Processing Functions ###############################
#                                                                              #
#  These Functions are meant to help you pre-process the data. File I/O is     #
#  done for you. create_attr_values and get_feature_vector are partially       # 
#  written for you, you will need to finish them for a woking solution.        #
#                                                                              #
#  Feel Free to add your own helper functions.                                 #
#                                                                              #
################################################################################
dict = [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]


# Input: Filename
# Output: List of lines read from that file
def read_data_file (filename):
  with open(filename, 'r') as f:
    data = [line for line in csv.reader(f)]

  return data


# Input: A list of strings with the first 15 values in an entry in the data file
# Output: A list of 15 floats contianing how you would 
def get_feature_vector(string_list):
    """ while you only need integeral values for the discrete attributes, you can
        represent them as floats to be able to simply represent the feature vector
        as a floats list

        Things you need to consider:
        - There are mising data entries "?" in all fields
        - continuous data is not always represented as float (for example it can
          be "3" instead of "3.0".
    """
    feature_vector = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    # You can group processing for continuous features:
    for i in [1,2,7,10,13,14]:
      if string_list[i] != "?":
        feature_vector[i] = float(string_list[i])
      else:
        feature_vector[i] = "?"

    # Then you can process discrete features.
    for i in [0,3,4,5,6,8,9,11,12]:
      value = string_list[i]
      if value != "?":
        if value in dict[i]:
          ref = dict[i][value]
          feature_vector[i] = ref
        else:
          if not dict[i]:
            dict[i][value] = 0
            feature_vector[i] = 0
          else:
            top = max(dict[i].values())
            dict[i][value] = top+1
            feature_vector[i] = top+1
      else:
        feature_vector[i] = "?"
          
    
    return feature_vector

# Input: A list of lines read from the data file
# Output: A list of 15 attr_ 
def process_data(dataset):

  # This model contains all our extracted data
  attributes = [(2,[]),(-1,[]),(-1,[]),(4,[]),(3,[]),
                (14,[]),(9,[]),(-1,[]),(2,[]),(2,[]),
                (-1,[]),(2,[]),(3,[]),(-1,[]),(-1,[])]
  labels = []
  
  for entry in dataset:
    # Extract label
    feature_vector = get_feature_vector(entry[0:15])

    # Get feature vector
    labels.append(entry[15])

    # Add your code here to populate data for each feature!
    for i in range(15):
        #What to do about missing data
        attributes[i][1].append(feature_vector[i])
    
  
  return (attributes, labels)


########################## Training Functions ##################################
#                                                                              #
#  These are the main functions that you will need to implement in order to    #
#  estimate the probability distributions. Each function is described in more  #
#  more detail in your handout.                                                #
#                                                                              #
#  Feel Free to add your own helper functions.                                 #
#                                                                              #
################################################################################

# Input: Attribute values (n,data) and given labels
# Output: a tuple of parameters for estimated probability distributions
def estimate_continuous(attr_values, labels):

  # This assumes you are returning a single parameter per estimation
  # how many parameters will you actually return?
  given_acc = [0.0,0.0]
  given_rej = [0.0,0.0]
  
  acc = []
  rej = []
  
  num = len(attr_values[1])

  # Here is where you need to compute your parameters
  for i in range(num):
    value = attr_values[1][i]
    if value != "?":
      if labels[i] == "+":
        acc.append(value)
      elif labels[i] == "-":
        rej.append(value)
      
        
  acc_mean = sum(acc)/len(acc)
  rej_mean = sum(rej)/len(rej)
  acc_var = sum(list(map(lambda x: (x-acc_mean)**2,acc)))/(len(acc)-1)
  rej_var = sum(list(map(lambda x: (x-rej_mean)**2,rej)))/(len(rej)-1)
  given_acc[0] = acc_mean
  given_acc[1] = acc_var
  given_rej[0] = rej_mean
  given_rej[1] = rej_var
    

  # change to immutable tuples
  estimate_given_acc = tuple(given_acc)
  estimate_given_rej = tuple(given_rej)
  
  return (estimate_given_acc, estimate_given_rej)

# Input: Attribute values (n,data) and given labels
# Output: a tuple of probability list where the ith element in each
#         list represents the conditional MLE of the ith value.
def estimate_discrete(attr_values, labels):
  
  given_acc = []
  given_ref = []
  posLabel = 0
  negLabel = 0
  
  n = attr_values[0]
  num = len(attr_values[1])
  
  for i in range(n):
    given_acc.append(0)
    given_ref.append(0)
  

  for i in range(num):
    # here you compute the log-probability that attr = ith value given each label
 
    label = labels[i]
    value = attr_values[1][i]
    #What if label is unknown?
    if value != "?":
      if label == "+":
        given_acc[value]+=1
        posLabel += 1
      elif label == "-":
        given_ref[value]+=1
        negLabel += 1
      
        
  for i in range(n):
    oldAcc = given_acc[i]
    oldRef = given_ref[i]
    if oldAcc != 0:
      given_acc[i] = oldAcc/(posLabel)
    else:
      given_acc[i] = 0
    if oldRef != 0:
      given_ref[i] = oldRef/(negLabel)
    else:
      given_ref[i] = 0

    

  # change to immutable tuples
  estimate_given_acc = tuple(given_acc)
  estimate_given_rej = tuple(given_ref)
  
  return (estimate_given_acc, estimate_given_rej)

# Input: list of labels as per the data file
# Output: probability that any application is accepted (given prior: Beta(7,9))
def probability_acc(labels):
  acc = len(list(filter(lambda x: x=="+",labels)))
  return (6+acc)/(len(labels)+14)


# Input: trained_model (tuple of 16 entries representing learnt estimations) and
#        X (a feature vector of 15 values)
# Output: (classification, probability X given acc, probability of X given rej
def estimate(trained_model, X):
  
  pacc = trained_model[15][0]
  prej = trained_model[15][1]
  probPlus = 1
  probMinus = 1

  
  for i in [1,2,7,10,13,14]:
    muPlus = trained_model[i][0][0]
    varPlus = trained_model[i][0][1]
    muMinus = trained_model[i][1][0]
    varMinus = trained_model[i][1][1]
   
    
    if X[i] != "?":
      xiGivenPlus = 1/(np.sqrt(2*np.pi*varPlus))*(np.e**(-((X[i]-muPlus)**2)/(2*varPlus)))
      xiGivenMinus = 1/(np.sqrt(2*np.pi*varMinus))*(np.e**(-((X[i]-muMinus)**2)/(2*varMinus)))
    
      probPlus = probPlus*xiGivenPlus
      probMinus = probMinus*xiGivenMinus


  # Then you can process discrete features.
  for i in [0,3,4,5,6,8,9,11,12]:
    #Assuming I see every value of a feature
    if X[i] != "?":
      probPlus = probPlus*trained_model[i][0][X[i]]
      probMinus = probMinus*trained_model[i][1][X[i]]
    
  if probPlus >= probMinus:
    #print("+")
    return ("+",probPlus,probMinus)
  else:
    #print("-")
    return ("-",probPlus,probMinus)
      
    
  

########################## Evaluation Function #################################
#                                                                              #
#  This function simply calculates classification error of your model given a  #
#  testing data file. The function is ready-to-use once you finish writin:     #
#       - get_feature_vector in Pre-processing section                         #
#       - estimate and all its dependencies                                    #
#                                                                              #
################################################################################
def ClassificationError(filename, model):
  data = read_data_file(filename)
  total_count = 0;
  error_count = 0;
  
  for entry in data:
    estimation = estimate(model, get_feature_vector(entry[0:15]))
    
    # Comparing the returned classification to the correct label
    if (entry[15] != estimation[0]):
    	error_count += 1

    total_count += 1 
    
  return (error_count*1.0/(total_count*1.0))
    
  
if __name__ == "__main__":
  
  ################### PRE-PROCESSING ###########################################
  training_data = read_data_file("data/training.dat")
  
  # 1. Extract the array of labels and attribute values from your training data
  (attributes, labels) = process_data(training_data)
  
  ################### TRAINING #################################################
  estimated_probabilities = []
  
  for attr_values in attributes:    
    # 3. Use attribute values to estimate its probabilistic distribution
    if (attr_values[0] < 0):
    	attr_probability = estimate_continuous(attr_values, labels)
    else: 
      attr_probability = estimate_discrete(attr_values, labels)

    estimated_probabilities.append(attr_probability)
      
  # 4. Estimate the class distributions
  p_acc = probability_acc(labels)
  print("Estimated Probability that an application is accepted: %f", p_acc)
  estimated_probabilities.append((p_acc, 1 - p_acc))

  # 5. Turning Your model into immutable tuple
  trained_model = tuple(estimated_probabilities)

  print(np.log(trained_model[0][0][1]))
  print(np.log(trained_model[0][1][1]))

  print(np.log(trained_model[5][0][4]))
  print(np.log(trained_model[5][1][4]))
  print(dict[5]["q"])
  
  ################## CLASSICATION ##############################################
  X1 = ["b","28.25","0.875","u","g","m","v",
        "0.96","t","t","03","t","g","396","0"]
  X2 = ["b","42.75","4.085","u","g","aa","v",
        "0.04","f","f","0","f","g","108","100"]
  X3 = ["a","46.08","3","u","g","c","v","2.375"
        ,"t","t","8","t","g","396","4159"]

  # 5. Test your work by classifying these three examples
  print("Classifcations of unlabeled applications:")
  print([estimate(trained_model, get_feature_vector(X1))[0],np.log(estimate(trained_model, get_feature_vector(X1))[1]),np.log(estimate(trained_model, get_feature_vector(X1))[2])])
  print([estimate(trained_model, get_feature_vector(X2))[0],np.log(estimate(trained_model, get_feature_vector(X2))[1]),np.log(estimate(trained_model, get_feature_vector(X2))[2])])
  print([estimate(trained_model, get_feature_vector(X3))[0],np.log(estimate(trained_model, get_feature_vector(X3))[1]),np.log(estimate(trained_model, get_feature_vector(X3))[2])])

  ################## EVALUATE ##################################################

  # 6. Uncomment this for classification errors once you are finished
  print("Error of training data:")
  print(ClassificationError('data/training.dat', trained_model))

  print("Error of testing data")
  print(ClassificationError('data/testing.dat', trained_model))
    
  
