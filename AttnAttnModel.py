import numpy as np
import tensorflow as tf
import DARNN
import newLSTM
import AttnAttn
import AttnAttn1


class Model():
    """
    모든 예측모델들의 기본 클래스
    """
    def __init__(self,sess,name,windowsize,stockNum):
        self.sess=sess
        self.name=name
        self.T=windowsize
        self.N=stockNum       

        self._build_net()

    def _build_net(self):
        #will be defined in subclass
        self.Y=None
        self.X=None
        self.Target=None
        self.a=None
        self.out=None
        self.cost=None
        self.optimizer=None
        self.batch=None
        

    def outputs(self,scaler,data,batch):
        X=[]
        Y=[]
        target=[]

        for d in data:
            X.append(d['X']);Y.append(d['Y']);target.append(d['target'])
        
        
        fd={self.Y:Y,self.X:X,self.Target:target,self.batch:batch}
        outputs=self.sess.run(self.out,feed_dict=fd)
        scaler_outputs=scaler.inverse_transform(outputs)
        scaler_target=scaler.inverse_transform(target)

        for i in range(50):#len(outputs)):
            print(outputs[i],scaler_outputs[i],'\ttarget:',target[i],scaler_target[i])

    def training(self,data,batch,a):
        X=[]
        Y=[]
        target=[]

        for d in data:
            X.append(d['X']);Y.append(d['Y']);target.append(d['target'])

        fd={self.Y:Y,self.X:X,self.Target:target,self.batch:batch,self.a:a}
        return self.sess.run([self.cost,self.optimizer],feed_dict=fd)

    def returnCost(self,data,batch):
        X=[]
        Y=[]
        target=[]

        for d in data:
            X.append(d['X']);Y.append(d['Y']);target.append(d['target'])

        fd={self.Y:Y,self.X:X,self.Target:target,self.batch:batch}
        return self.sess.run(self.cost,feed_dict=fd)

class LSTM_Model(Model):
    """
    Prediction model for test.
    """
    def _build_net(self):
        self.Y=tf.placeholder(tf.float32,[None,self.T,1])
        self.X=tf.placeholder(tf.float32,[None,self.T,self.N])
        self.Target=tf.placeholder(tf.float32,[None,1])
        self.a = tf.placeholder(tf.float32)
        self.batch=tf.placeholder(tf.int32)
                
        
        LSTM=tf.nn.rnn_cell.LSTMCell(1)
        out,_=tf.nn.dynamic_rnn(LSTM,self.Y,dtype=tf.float32)

        self.out=tf.layers.flatten(out)
        
        self.cost=tf.pow(tf.reduce_mean(tf.square(tf.subtract(self.out,self.Target))),0.5)
        #self.cost=tf.pow(tf.losses.mean_squared_error(labels=self.Target,predictions=self.out),0.5)
        self.optimizer=tf.train.AdamOptimizer(self.a).minimize(self.cost)


class MILSTM_Model(Model):
    """
    Prediction model for test.
    """
    def _build_net(self):
        self.Y=tf.placeholder(tf.float32,[None,self.T,1])
        self.X=tf.placeholder(tf.float32,[None,self.T,self.N])
        self.Target=tf.placeholder(tf.float32,[None,1])
        self.a = tf.placeholder(tf.float32)
        self.batch=tf.placeholder(tf.int32)
                

    
        LSTM=tf.nn.rnn_cell.LSTMCell(64,name='lstm1')
        
        Y_1,_=tf.nn.dynamic_rnn(LSTM,self.Y,dtype=tf.float32)        
        Xlist=tf.split(self.X,self.N,axis=2)

        xresult=[]
        for x in Xlist:
            o,_=tf.nn.dynamic_rnn(LSTM,x,dtype=tf.float32)
            xresult.append(o)

        #x_result=tf.concat(xresult,2)
        x_result=tf.reduce_mean(xresult,axis=0)
        #print(tf.shape(x_result))


        X=tf.concat([Y_1,x_result],2)
        #MI-LSTM
        LSTM2=newLSTM.MI_LSTMCell(64,2,name='lstm2')
        Y_2,_ =tf.nn.dynamic_rnn(LSTM2,X,dtype=tf.float32)

        #Attention_Layer
        attention_layer=newLSTM.Attention_Layer(self.T,64)
        Y_3=attention_layer(Y_2)

        #Non-linear units for producing final prediction.
        R_1=tf.layers.dense(tf.layers.flatten(Y_3),64,tf.nn.relu)
        R_6=tf.layers.dense(R_1,1)

        self.out=R_6
        
        self.cost=tf.pow(tf.losses.mean_squared_error(labels=self.Target,predictions=self.out),0.5)
        self.optimizer=tf.train.AdamOptimizer(self.a).minimize(self.cost)


class MILSTM_Model_PN(Model):
    """
    Prediction model for test.
    """
    def training(self,data,batch,a):
        P=[]
        N=[]
        Y=[]
        target=[]

        for d in data:
            P.append(d['P']);N.append(d['N']);Y.append(d['Y']);target.append(d['target'])

        fd={self.Y:Y,self.P:P,self.N:N,self.Target:target,self.batch:batch,self.a:a}
        return self.sess.run([self.cost,self.optimizer],feed_dict=fd)

    def returnCost(self,data,batch):
        P=[]
        N=[]
        Y=[]
        target=[]

        for d in data:
            P.append(d['P']);N.append(d['N']);Y.append(d['Y']);target.append(d['target'])

        fd={self.Y:Y,self.P:P,self.N:N,self.Target:target,self.batch:batch}
        return self.sess.run(self.cost,feed_dict=fd)

    def _build_net(self):
        self.Y=tf.placeholder(tf.float32,[None,self.T,1])
        #self.X=tf.placeholder(tf.float32,[None,self.T,self.N])
        self.P=tf.placeholder(tf.float32,[None,self.T,10])
        self.N=tf.placeholder(tf.float32,[None,self.T,10])
        self.Target=tf.placeholder(tf.float32,[None,1])
        self.a = tf.placeholder(tf.float32)
        self.batch=tf.placeholder(tf.int32)
                

    
        LSTM=tf.nn.rnn_cell.LSTMCell(64,name='lstm1')
        
        Y_1,_=tf.nn.dynamic_rnn(LSTM,self.Y,dtype=tf.float32)        
        Plist=tf.split(self.P,10,axis=2)
        Nlist=tf.split(self.N,10,axis=2)

        presult=[]
        for x in Plist:
            o,_=tf.nn.dynamic_rnn(LSTM,x,dtype=tf.float32)
            presult.append(o)
        presult=tf.concat(presult,2)

        nresult=[]
        for x in Nlist:
            o,_=tf.nn.dynamic_rnn(LSTM,x,dtype=tf.float32)
            nresult.append(o)
        nresult=tf.concat(nresult,2)

        X=tf.concat([Y_1,presult,nresult],2)
        #MI-LSTM
        LSTM2=newLSTM.MI_LSTMCell(64,3,name='lstm2')
        Y_2,_ =tf.nn.dynamic_rnn(LSTM2,X,dtype=tf.float32)

        #Attention_Layer
        attention_layer=newLSTM.Attention_Layer(self.T,64)
        Y_3=attention_layer(Y_2)

        #Non-linear units for producing final prediction.
        R_1=tf.layers.dense(tf.layers.flatten(Y_3),64,tf.nn.relu)
        R_6=tf.layers.dense(R_1,1)

        self.out=R_6
        
        self.cost=tf.pow(tf.losses.mean_squared_error(labels=self.Target,predictions=self.out),0.5)
        self.optimizer=tf.train.AdamOptimizer(self.a).minimize(self.cost)

class DARNN_Model(Model):
    """
    Prediction model for test.
    """
    def _build_net(self):
        self.Y=tf.placeholder(tf.float32,[None,self.T,1])
        self.X=tf.placeholder(tf.float32,[None,self.T,self.N])
        self.Target=tf.placeholder(tf.float32,[None,1])
        self.a = tf.placeholder(tf.float32)
        self.batch=tf.placeholder(tf.int32)        

        InputLSTM=DARNN.InputAttention_LSTMCell(64,self.N,self.T,self.X)
        
        en_states,_=tf.nn.dynamic_rnn(InputLSTM,self.X,dtype=tf.float32)

        TempoLSTM=DARNN.TemporalAttention_LSTMCell(64,1,self.T,64,en_states)

        _,result=tf.nn.dynamic_rnn(TempoLSTM,self.Y,initial_state=tf.zeros([self.batch,192]))
        c, h, cont = tf.split(result,3,1)

        y = tf.concat([h,cont],1)

        R_6=tf.layers.dense(y,1)

        self.out=R_6
        
        self.cost=tf.pow(tf.losses.mean_squared_error(labels=self.Target,predictions=self.out),0.5)
        self.optimizer=tf.train.AdamOptimizer(self.a).minimize(self.cost)

class Att2d_Model_BEST(Model):
    """
    Prediction model for test.
    """
    def _build_net(self):
        self.Y=tf.placeholder(tf.float32,[None,self.T,1])
        self.X=tf.placeholder(tf.float32,[None,self.T,self.N])
        self.Target=tf.placeholder(tf.float32,[None,1])
        self.a = tf.placeholder(tf.float32)
        self.batch=tf.placeholder(tf.int32)
                
        DAL1=AttnAttn1._2DAttnLayer(24,self.N,self.T,self.X,self.Y,self.batch)
        DAL2=AttnAttn1._2DAttnLayer(24,DAL1.output_dimension,self.T,DAL1(),self.Y,self.batch)
        DAL3=AttnAttn1._2DAttnLayer(24,DAL2.output_dimension,self.T,DAL2(),self.Y,self.batch)
        #DAL4=AttnAttn1._2DAttnLayer(16,DAL3.output_dimension,self.T,DAL3(),self.Y,self.batch)
        #DAL5=AttnAttn1._2DAttnLayer(16,DAL4.output_dimension,self.T,DAL4(),self.Y,self.batch)
        #DAL6=AttnAttn1._2DAttnLayer(16,DAL5.output_dimension,self.T,DAL5(),self.Y,self.batch)
        PL=tf.layers.Dense(1)
        #attention_layer=newLSTM.Attention_Layer(self.T,64)
        #Y_=attention_layer(DAL1())

        

        DAL4_r=DAL3()[:,-1]

        self.out=PL(tf.layers.flatten(DAL4_r))
        
        self.cost=tf.pow(tf.reduce_mean(tf.square(tf.subtract(self.out,self.Target))),0.5)
        #self.cost=tf.pow(tf.losses.mean_squared_error(labels=self.Target,predictions=self.out),0.5)
        self.optimizer=tf.train.AdamOptimizer(self.a).minimize(self.cost)





########################parameter setting
class Att2d_Model1_(Model):
    """
    Prediction model for test.
    """
    def __init__(self,sess,name,windowsize,stockNum,M,L):
        self.sess=sess
        self.name=name
        self.T=windowsize
        self.N=stockNum       

        self.M = M #hidden unit
        self.L = L #layer count
        self._build_net()
        
    def _build_net(self):
        self.Y=tf.placeholder(tf.float32,[None,self.T,1])
        self.X=tf.placeholder(tf.float32,[None,self.T,self.N])
        self.Target=tf.placeholder(tf.float32,[None,1])
        self.a = tf.placeholder(tf.float32)
        self.batch=tf.placeholder(tf.int32)

        DAL_list=[]   
        """   
        DAL1=AttnAttn1._2DAttnLayer(self.M,self.N,self.T,self.X,self.Y,self.batch)
        DAL2=AttnAttn1._2DAttnLayer(self.M,DAL1.output_dimension,self.T,DAL1(),self.Y,self.batch)
        DAL3=AttnAttn1._2DAttnLayer(self.M,DAL2.output_dimension,self.T,DAL2(),self.Y,self.batch)
        DAL4=AttnAttn1._2DAttnLayer(self.M,DAL3.output_dimension,self.T,DAL3(),self.Y,self.batch)
        DAL5=AttnAttn1._2DAttnLayer(self.M,DAL4.output_dimension,self.T,DAL4(),self.Y,self.batch)
        DAL6=AttnAttn1._2DAttnLayer(self.M,DAL5.output_dimension,self.T,DAL5(),self.Y,self.batch)
        """
        DAL_list.append(AttnAttn1._2DAttnLayer(self.M,self.N,self.T,self.X,self.Y,self.batch))
        for i in range(1,self.L):
            DAL_list.append(AttnAttn1._2DAttnLayer(self.M,DAL_list[-1].output_dimension,self.T,DAL_list[-1](),self.Y,self.batch))
        PL=tf.layers.Dense(1)

        DAL4_r=DAL_list[-1]()[:,-1]

        self.out=PL(tf.layers.flatten(DAL4_r))
        
        self.cost=tf.pow(tf.reduce_mean(tf.square(tf.subtract(self.out,self.Target))),0.5)
        #self.cost=tf.pow(tf.losses.mean_squared_error(labels=self.Target,predictions=self.out),0.5)
        self.optimizer=tf.train.AdamOptimizer(self.a).minimize(self.cost)


