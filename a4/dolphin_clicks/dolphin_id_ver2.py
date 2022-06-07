# Might make your life easier for appending to lists
from collections import defaultdict

# Third party libraries
import numpy as np
# Only needed if you plot your confusion matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import regularizers 
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

# our libraries
from lib.partition import split_by_day
import lib.file_utilities as util
from lib.buildmodels import build_model

# Any other modules you create


def dolphin_classifier(data_directory):
    """
    Neural net classification of dolphin echolocation clicks to species
    :param data_directory:  root directory of data
    :return:  None
    """
    
    plt.ion()   # enable interactive plotting

    use_onlyN = np.Inf # np.Inf # debug, only read this many files for each species
    
    split_functions = [split_by_day, split_by_location]

    for func in split_functions: 

        #split data by specified function
        train, test = get_data(data_directory,use_onlyN, func) 

        #get training annd testing data in correct dimensions
        X_train, Y_train, X_test, Y_test = prepare_data(train, test) 

        #define class weight 
        class_weight = compute_class_weight(Y_train)

        print(X_train.shape)
        print(Y_train.shape)
        print(X_test.shape)
        print(Y_test.shape)

        x = 0
        y = 0
        for i in Y_train:
            if i == 0: 
                x += 1
            else: 
                y+=1

        print(x,y)

        # model = Sequential()

        i_size = len(X_train[0]) #no. of features
        nodes = 100 #no. of neurons
        categories = 2 #output dimension, since we have two categories

        #build the model
        model = build_model([ (Input, [], {'shape':(i_size,)}),
        (Dense, [nodes], {'activation':'relu', 'kernel_regularizer':regularizers.L2(0.01)}),
        (Dense, [nodes], {'activation':'relu', 'kernel_regularizer':regularizers.L2(0.01)}),
        (Dense, [nodes], {'activation':'relu', 'kernel_regularizer':regularizers.L2(0.01)}),
        (Dense, [categories], {'activation':'softmax'})])

        # model.add(Input(shape=(i_size,)))
        # model.add(Dense(nodes, activation='relu', kernel_regularizer=regularizers.L2(0.01)))
        # model.add(Dense(nodes, activation='relu', kernel_regularizer=regularizers.L2(0.01)))
        # model.add(Dense(nodes, activation='relu', kernel_regularizer=regularizers.L2(0.01)))
        # model.add(Dense(categories, activation='softmax'))

        #model.summary()
        #optimization: stochastic gradient descent: SGD, batch gradient descent 
        model.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

        Y_onehot_train = to_categorical(Y_train) #one-hot labels

        #train the model
        model.fit(X_train, Y_onehot_train, batch_size=100, epochs=10, class_weight=class_weight)

        #predict in groups of 100 and leave out the last group if it is an incomplet set 
        cycles = int(len(X_test)/100)*100 

        Y_pred_new = []
        
        #get one hot labels for grouping 
        Y_onehot_test = to_categorical(Y_test)
        Y_test_new = []

        for i in range(0,cycles,100): 
            #to compute the predicted label for every 100 clicks
            Y_pred = model.predict(X_test[i:i+99])
            joint_prob = np.sum(np.log(Y_pred),axis=0) #sum the log of probabilities
            Y_pred_new.append(np.argmax(joint_prob) )#decide which label it should be for these 100 clicks
            
            #to group Y_test in to groups of 100 for comparison with Y_pred_new so that they have the same dimensions
            labels = np.sum(Y_onehot_test[i:i+99],axis=0)
            Y_test_new.append(np.argmax(labels))

        accuracy, error_rate = confusion_matrix_plot(Y_test_new, Y_pred_new, func)
        
        print("Accuracy: {:.2%}, Error rate: {:.2%}\n".format(accuracy, error_rate))

    
def get_data(data_directory, use_onlyN, split_function): 
    """
    This function gets the data from the directories and splits the data according to the specified split function separately
    (One for Risso's dolphin and one for Pacific white-sided dolphin). 
    It then splits the data for training(70%) and testing(30%).
    All this data is then compiled into 1 set of training data and 1 set of testing data.

    :param data_directory:  root directories of data 
    :param use_onlyN: for debugging purposes to limit data read in 
    :param split_function: function used to split the data, either by day or by location

    :return:  List of training data, List of testing data 
    """
    
    #get Risso's dolphin clicks and split into training and test data
    Risso_files = util.get_files(data_directory[0], stop_after = use_onlyN)
    Risso_tuple = util.parse_files(Risso_files) #organize data into a tuple
    Risso_dict = split_function(Risso_tuple) #split data according to specified split function
    Risso_split = list(Risso_dict.keys()) #convert dictionary keys to list
    Risso_split_train, Risso_split_test = train_test_split(Risso_split,random_state=42) #split data into training and test data 
    
    #get Pacific white-sided dolphin clicks and split into training and test data
    Pacific_files = util.get_files(data_directory[1], stop_after = use_onlyN)
    Pacific_tuple = util.parse_files(Pacific_files)
    Pacific_dict = split_function(Pacific_tuple) 
    Pacific_split = list(Pacific_dict.keys())
    Pacific_split_train, Pacific_split_test = train_test_split(Pacific_split,random_state=42) #split data into training and test data 

    #Gather all lists for training and testing into one list 
    train = []
    test = []
    for split_var in Risso_split_train: 
        for i in range(len(Risso_dict[split_var])):
            train.append(list(Risso_dict[split_var][i]))
    for split_var in Risso_split_test: 
        for i in range(len(Risso_dict[split_var])):
            test.append(list(Risso_dict[split_var][i]))
    
    for split_var in Pacific_split_train: 
        for i in range(len(Pacific_dict[split_var])):
            train.append(list(Pacific_dict[split_var][i]))
    for split_var in Pacific_split_test: 
        for i in range(len(Pacific_dict[split_var])):
            test.append(list(Pacific_dict[split_var][i]))
    
    return train, test


def prepare_data(train, test):
    """
    This function prepares the data in the correct dimensions to be used for the model
    Examples tensors(X_train, X_test): Nx20 (N clicks by 20 features)
    Labels tensor(Y_train, Y_test): N 

    :param train:  List of training data 
    :param test: List of testing data 

    :return:  Tensors of Training examples, Training labels, Testing examples, Testing labels
    """

    X_train = train[0][3]
    Y_train = []
    
    # training set
    for i in range(len(train)): 
        if i != 0: 
            X_train = np.vstack((X_train, np.array(train[i][3]))) 

        if train[i][1] == "Gg":  #Risso's Dophin is 0
            for j in range(len(train[i][3])):
                Y_train.append(0) 
        else:                    #Pacific white-sided dolphin is 1
            for j in range(len(train[i][3])):               
                Y_train.append(1)
    
    # test set 
    X_test = test[0][3]
    Y_test = []
    for i in range(len(test)):
        if i != 0: 
            X_test = np.vstack((X_test, np.array(test[i][3])))

        if test[i][1] == "Gg":  #Risso's Dophin is 0
            for j in range(len(test[i][3])):
                Y_test.append(0) 
        else:                   #Pacific white-sided dolphin is 1
            for j in range(len(test[i][3])):
                Y_test.append(1)
    
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    
    return X_train, Y_train, X_test, Y_test

def compute_class_weight(Y_train):
    """
    This function computes the class weight to balance the data.

    :param Y_train:  List of training labels

    :return: a dictionary of the computed class weight
    """
    
    class_weight = defaultdict()

    Risso = len(Y_train)-np.sum(Y_train)
    Pacific = np.sum(Y_train)

    if Risso > Pacific: 
        class_weight[0] = 1
        class_weight[1] = Risso/Pacific
    else: 
        class_weight[0] = Pacific/Risso
        class_weight[1] = 1
    
    return class_weight


def confusion_matrix_plot(Y_test, Y_pred, func): 
    """
    This function plots the confusion matrix and calculates the accuracy and error rate of the model.
    confusion_matrix returns a matrix in the form:  
    [[TP, FN]
     [FP, TN]]

    :param Y_test:  List of actual labels
    :param Y_pred: List of predicted labels
    :param Y_pred: Split function used 

    :return:  The accuracy and the error rate of the models computed from the confusion matrix 
    """
    
    cm = confusion_matrix(Y_test,Y_pred)

    #accuracy = (TP+TN)/P+N
    accuracy = np.trace(cm) / np.sum(cm).astype('float') 
    error_rate = 1 - accuracy
    
    classes = ['Gg','Lo']
    tick_marks = np.arange(len(classes))
    title = '{} Confusion Matrix for Dolphin Classification'.format(func.__name__)

    #matrix display
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    
    #insert confusion matrix values 
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment="center", size = 'xx-large')
    
    plt.title(title)

    #labels for dolphin class
    plt.xticks(tick_marks, classes) 
    plt.yticks(tick_marks, classes) 

    #axis labels
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class\naccuracy={:0.4f}; error rate={:0.4f}'.format(accuracy, error_rate))
    
    filename = "{}_Confusion_Matrix.png".format(func.__name__) 
    plt.savefig(filename, bbox_inches = 'tight')

    #plt.colorbar()
    #plt.tight_layout()
    #plt.show()
    #insert confusion matrix values 
    # thresh = cm.max() / 2.
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         plt.text(j, i, cm[i, j], 
    #                  horizontalalignment="center", 
    #                  size = 'xx-large', 
    #                  color="white" if cm[i, j] > thresh else "black")

    return accuracy, error_rate


def split_by_location(recordings):
    """
    Given a list of RecordingInfoTypes, split them into a list of lists where
    each list represents one site/location of recording.
    :param recordings:
    :return: A list where the data is split by location
    """

    bylocation = defaultdict(list)
    for r in recordings:
        # For each recording, append to a list keyed on the recording site
        site = r.site
        bylocation[site].append(r)

    return bylocation


if __name__ == "__main__":
    data_directory = ["/Users/wwe/Documents/SDSU Classes/Senior/Fall 2021/CS550 - Artificial Intelligence/a4/features/Gg", "/Users/wwe/Documents/SDSU Classes/Senior/Fall 2021/CS550 - Artificial Intelligence/a4/features/Lo"] #"path\to\data"  # root directory of data
    dolphin_classifier(data_directory)