import os
import pickle
import pandas as pd
import datetime
import calendar
from dateutil.relativedelta import relativedelta
from collections import deque
from sklearn import preprocessing  
import random
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras 

EPOCHS = 10
BATCH_SIZE = 64
SEQ_LEN = 12  # how long of a preceeding sequence to collect for RNN sequence --> 6 months
FUTURE_PERIOD_PREDICT = 1 # predict 1 months in the future
PERCENT_VALIDATION_DATA = 0.15 # 15 percent is used towards testing
NAME = "STOCK-PREDICTION-{}-{}-SEQ-{}".format(SEQ_LEN,FUTURE_PERIOD_PREDICT,int(time.time()))  # a unique name for the model
RETREIVE_FILE_NAME_LST = False # call retreive_file_name_data and create a pickle

def preprocess_df(splt_df):
    ''' scales and balances the input data '''
    
    ticker = splt_df.columns[0].split('_')[0]
    splt_df = splt_df.drop("{}_Future".format(ticker), 1)  # don't need this anymore.
    
    for col in splt_df.columns:  
        try:
            if col != "{}_Target".format(ticker):  # normalize all ... except for the target itself!
                splt_df[col] = splt_df[col].pct_change()  
                splt_df.dropna(inplace=True)  # remove the nas created by pct_change
                splt_df[col] = preprocessing.scale(splt_df[col].values)  # scale between 0 and 1.
        except ValueError:
            assert "Error in Pandas Processing"
        
    splt_df.dropna(inplace=True)
    
    sequential_data = []  # this is a list that will CONTAIN the sequences
    days = SEQ_LEN * 30 # approximate conversion to number of days. Ensures consistent input shape
    prev_days = deque(maxlen=days)  # These will be our actual sequences.

    for i in splt_df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the Future price
        if len(prev_days) == days: 
            sequential_data.append([np.array(prev_days), i[-1]]) 
            
    random.shuffle(sequential_data)  # shuffle for good measure.

    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = buys+sells  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.
    
    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!

# Temporary Function
def classify(current, future):
    if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0

def split_df(df):
    ''' splits the df into validation and training dfs '''
    times = sorted(df.index.values)
    val_sample = sorted(df.index.values)[-int(PERCENT_VALIDATION_DATA*len(times))] 
    validation_df = df[(df.index >= val_sample)].copy() # split training and testing data into it's split 
    train_df = df[(df.index < val_sample)].copy()
    
    return train_df,validation_df

def retreive_file_name_data():
    ''' finds the folder of stock information
    returns ETF_lst and stock_lst containing file names'''
    stock_file_name_lst = []
    ETF_file_name_lst = []
    for root, _, files in os.walk(".", topdown=False):
        if root.strip('./') == 'ETFs':
            for name in files:
                ETF_file_name_lst.append(name)
        elif root.strip('./') == 'Stocks':         
            for name in files:
                stock_file_name_lst.append(name)
    ETF_file_name_lst.sort()
    stock_file_name_lst.sort()
    return ETF_file_name_lst, stock_file_name_lst

def create_pickle(*args):
    ''' creates a pickle based on args passed, accepts a tuple (fl_lst, Name) '''
    for file in args:
        with open(file[1],"wb") as f_pick: 
            pickle.dump(file[0], f_pick)
    
def load_pickle(*args):
    ''' loads a pickle based on args passed, accepts file names (fl_name) '''
    return_data = []
    for file in args:
        with open(file+".pickle","rb") as f_pick: 
            data = pickle.load(f_pick)
            return_data.append(data)
    return return_data

def create_dataframe(f):
    ''' returns df containing Close and Volume columns '''
    
    dataset = "Stocks/{}".format(f)  # get the full path to the file.
    df = pd.read_csv(dataset)  # read in specific file
    ticker = f.split('.')[0]

    df.rename(columns={"Close": "{}_Close".format(ticker),
    "Volume": "{}_Volume".format(ticker)}, inplace=True) # rename volume and close to include the ticker

    df.set_index("Date", inplace=True)  # set time as index so we can join them on this shared time
    df = df[["{}_Close".format(ticker), "{}_Volume".format(ticker)]]  # ignore the other columns besides price and volume

    df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
    df.dropna(inplace=True)
    
    return df

def calculate_days(date,months):
    ''' return the number of days from date to months number of months '''
    year, month, day = date.split('-')
    cur_date, tar_date = datetime.date(int(year), int(month), int(day)), datetime.date(int(year), int(month), int(day)) + relativedelta(months=+months)
    days = (tar_date - cur_date).days
    return days

def create_target_column(df):
    ''' creates the target column based on future_period_predict through mutating df '''
    ticker = df.columns[0].split('_')[0]
    print("Processing Future column of {}".format(ticker))
    for date in df.index:
        try:
            index_shift = calculate_days(date,FUTURE_PERIOD_PREDICT)
            df['{}_Future'.format(ticker)] = df['{}_Close'.format(ticker)].shift(-index_shift)
            
            # do some mapping for an initial binary crossentropy
            df['{}_Target'.format(ticker)] = list(map(classify, df['{}_Close'.format(ticker)], df['{}_Future'.format(ticker)]))
        except KeyError:
            continue

def setup_model(shape):
    ''' creates the Sequential model and Tensorboard '''
    
    model = keras.Sequential()
    model.add(keras.layers.CuDNNLSTM(528, input_shape=(shape), return_sequences=True))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.BatchNormalization())  
    
    model.add(keras.layers.CuDNNLSTM(128))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Dense(1028))
    act = keras.layers.PReLU(alpha_initializer='zeros')
    model.add(act)
    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    
    tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(NAME))
    filepath = "RNN_Final-{epoch:02d}-{val_loss:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = keras.callbacks.ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) 
    
    model.summary()
    
    return model, tensorboard, checkpoint

def train_model(train_x, train_y, validation_x, validation_y, checkpoint, tensorboard, model):
    ''' trains the model based on the supplied tensors using train_on_batch
        train_model(np.array, np.array, keras.Sequential) --> History '''
    history = model.fit (
        train_x, train_y,
        batch_size=BATCH_SIZE,
        validation_data=(validation_x, validation_y),
        epochs=EPOCHS,
        callbacks=[checkpoint, tensorboard],
    )
    return history

def evaluate_model(validation_x, validation_y, model):
    ''' evaluates the performance of the model, and prints to terminal
        test_model(np.array, np.array, keras.Sequential) --> None '''

    score = model.evaluate(validation_x, validation_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if RETREIVE_FILE_NAME_LST: # saves the list as a pickle
    ETF_file_name_lst, stock_file_name_lst = retreive_file_name_data()
    create_pickle((ETF_file_name_lst,"ETF_file_name_lst"),(stock_file_name_lst,"stock_file_name_lst"))
else:
    file_name_lst = load_pickle("ETF_file_name_lst","stock_file_name_lst") # pack the file lists in a list
    ETF_file_name_lst,stock_file_name_lst = file_name_lst[0], file_name_lst[1] # unpack

train_stocks = stock_file_name_lst[:int(PERCENT_VALIDATION_DATA*(len(stock_file_name_lst)))]
validate_stocks = stock_file_name_lst[int(PERCENT_VALIDATION_DATA*(len(stock_file_name_lst))):]

for index, (train_f,validate_f) in enumerate(zip(train_stocks,validate_stocks)): # train the model

    try:
        train_df,validation_df = create_dataframe(train_f),create_dataframe(validate_f)
    except pd.errors.EmptyDataError:
        print("file error")
        continue
               
    _,_ = create_target_column(train_df),create_target_column(validation_df)
    
    train_x, train_y = preprocess_df(train_df)
    validation_x, validation_y = preprocess_df(validation_df) 

    if index == 0: # if the model doesn't exist
        model, tensorboard, checkpoint = setup_model(train_x.shape[1:])

    history = train_model(train_x, train_y, validation_x, validation_y, checkpoint, tensorboard, model) # fit the model

    evaluate_model(validation_x, validation_y, model) # evaluate the performance of the model

# Save model
model.save("models/{}".format(NAME))
