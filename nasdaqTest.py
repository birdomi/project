import numpy as np
import tensorflow as tf
import nasdaqData as data
import AttnAttnModel as model
import time
import itertools

#parameter list
nasdaq=data.NasdaqData('nasdaq/nasdaq100_padding.csv')
name='lstm'
timesize=nasdaq.T
batch_size=100000
#filename = 'result/03_19_nasdaq100_.txt'
filename = 'result/03_22_FIANL_test.txt'
#

#print('종목개수 : ',kospi.N, '데이터셋 수: ',kospi.dataCount)
print('\n#training#')


re=[]
for q in range(5):
    sess=tf.Session()
    selectedModel=model.Att2d_Model_BEST(sess,'Att2d',timesize,nasdaq.exoNum)
    sess.run(tf.global_variables_initializer())
    result_dic={}
    min_cost = 10000
    epoch = 0
    n = 0
    learning_rate = 0.01

    while(n<30):
        epoch += 1
        if(epoch%10000 is 0):
            learning_rate=learning_rate*0.9
        nasdaq.shuffle()

        start_time = time.time()
        training_cost=0
        validation_cost=0
        evalution_cost=0

        for minibatch in nasdaq.getBatch('training'):
            c,_=selectedModel.training(minibatch,nasdaq.batchSize,learning_rate)
            training_cost+=c
       
        validation_cost=selectedModel.returnCost(nasdaq.validationSet,len(nasdaq.validationSet))
        evalution_cost=selectedModel.returnCost(nasdaq.testingSet,len(nasdaq.testingSet))

        training_cost /= nasdaq.batchNum['training']

        elapsed_time = time.time()-start_time            
        n+=1
        if(min_cost>evalution_cost):
            min_cost = evalution_cost
            n=0            
        result_dic[epoch]=[training_cost,validation_cost,evalution_cost]        

        print('epoch :{:>3}, t_cost : {:0.6f}, v_cost : {:0.6f}, e_cost : {:0.6f}, elapsed time : {:0.2f}sec'.format(
                epoch,training_cost,validation_cost,evalution_cost,elapsed_time))
    
        #
    sorted_result=sorted(result_dic,key=lambda k:result_dic[k][2])
    bestEpoch=sorted_result[0]
    print('\n#Best result at epoch {}'.format(bestEpoch))
    print('t_cost : {:0.6f}, e_cost : {:0.6f}'.format(result_dic[bestEpoch][0],result_dic[bestEpoch][2]))
    re.append(result_dic[bestEpoch][2])
    tf.reset_default_graph()    
print(re)

avg = np.mean(re)
best = np.min(re)

f = open(filename,mode='a')
f.write(selectedModel.name)
f.write(str(re))
f.write('avg : {}'.format(str(avg)))
f.write('\n\n')
f.close()