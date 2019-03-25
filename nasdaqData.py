import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from pandas_datareader import data,wb
import random

class NasdaqData():
    def __init__(self,csvfile):          
        self.csvfile=csvfile
        self.scaler = MinMaxScaler()
        self.T=10
        self.exoNum=0
        self.div=5
        self.batchSize=1024#1024

        self.trainingSet=[]
        self.testingSet=[]
        self.validationSet=[]

        self.batchNum={}

        i,s,t=self.loadCSV()
        self.createDataset(i,s,t)

    def loadCSV(self):
        """
        csv파일이 있는 폴더를 입력으로 받아 데이터를 읽어옴.
        """
        data=pd.read_csv(self.csvfile,engine='python')
        target_series=np.reshape(data['NDX'].values,[-1,1])
        exogenous = data.iloc[:,:-1].values
        exogenous=minmax_scale(exogenous)
        target_series=self.scaler.fit_transform(target_series)
        #print(target_series.shape)
        #print(exogenous.shape,exogenous)
        

        length=target_series.shape[0]
        self.exoNum=exogenous.shape[1]

        index=[]
        stock=[]
        target=[]
        for i in range(length-self.T):            
            index.append(target_series[i:i+self.T])
            stock.append(exogenous[i:i+self.T])
            target.append(target_series[i+self.T])
        index=np.reshape(index,[-1,self.T,1])
        stock=np.reshape(stock,[-1,self.T,self.exoNum])
        target=np.reshape(target,[-1,1])

        #print(index.shape,stock.shape,target.shape)
        return index,stock,target

    def createDataset(self,index,stock,target):
        setcount=int(len(index)/20)       
                            
        trainingRange = range(0,setcount*14) #70%
        validationRange = range(setcount*14,setcount*17) #15%
        testingRange = range(setcount*17,len(index)) #15%
        for i in trainingRange:                
            data={'X':stock[i],'Y':index[i],'target':target[i]}
            self.trainingSet.append(data)
        for i in validationRange:                
            data={'X':stock[i],'Y':index[i],'target':target[i]}
            self.validationSet.append(data)
        for i in testingRange:                
            data={'X':stock[i],'Y':index[i],'target':target[i]}
            self.testingSet.append(data)
        
        print(len(self.trainingSet),len(self.validationSet),len(self.testingSet))

    def shuffle(self):
        random.shuffle(self.trainingSet)

    def getBatch(self,option):
        """
        클래스에 저장된 세트에 대한 minibatch리턴.

        args:
            option='training' or 'evaluation' or 'validation'

        returns:
            batch 제너레이터
            batch={'X','Y','target'}
        """
        if(option is not 'training' and option is not 'evaluation' and option is not 'validation'):
            raise ValueError('option should be "training" or "evaluation" or "validation".')

        if(option is 'training'):
            returnSet = self.trainingSet
        elif(option is 'evaluation'):
            returnSet = self.testingSet
        else:
            returnSet = self.validationSet        
                
        batchNum=int(len(returnSet)/self.batchSize)
        self.batchNum[option]=batchNum

        for i in range(batchNum):
            yield returnSet[i*self.batchSize:(i+1)*self.batchSize]


nasdaq = NasdaqData('nasdaq/nasdaq100_padding.csv')
