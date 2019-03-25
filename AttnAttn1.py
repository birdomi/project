import tensorflow as tf
import DARNN
import numpy
rnn = tf.nn.rnn_cell
#InputCNN=tf.keras.layers.Conv2D(1,[1,3],1,padding='same',kernel_initializer=tf.ones_initializer)

class _2DAttnLayer():
    """
    CNN+InputAttn+TempAttn
    입력된 [-1,input_dimension,input_timesteps] 데이터를 [-1,input_dimension,LSTM_units]으로 출력.
    입력된 데이터에 대해 CNN -> InputAttn -> TemporalAttn순으로 계산.
    

    """
    def __init__(self,
                LSTM_units,
                input_dimension,
                input_timesteps,
                inputs,
                target_inputs,
                batch,
                name=None):
        self.output_dimension=LSTM_units

        inputCNN=tf.keras.layers.Conv2D(1,[1,5],1,padding='same')
        cnnResult=inputCNN(tf.reshape(inputs,[-1,input_timesteps,input_dimension,1]))
        cnnResult=tf.squeeze(cnnResult,axis=3)

        self.AttnCell=Attn_2D_Cell(LSTM_units,input_dimension,input_timesteps,cnnResult,inputs)
        self.result,_=tf.nn.dynamic_rnn(self.AttnCell,target_inputs,initial_state=tf.zeros([batch,LSTM_units*3]))

    def __call__(self):
        return self.result

class Attn_2D_Cell(rnn.BasicLSTMCell):
    """
    MY LSTM proposed.
    """
    def __init__(self,
               num_units,
               num_inputs,
               num_timesteps,
               CNN_states,
               input_states,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs):
        """
        Initialize the basic LSTM cell.
        """        
        super(Attn_2D_Cell,self).__init__(num_units,
               forget_bias=1.0,
               state_is_tuple=False,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs)
        self.inputs=input_states
        self.timesteps=num_timesteps

        self.CNNinputs_for_IA=tf.split(CNN_states,num_inputs,2)
        self.CNNinputs_for_TA=tf.split(CNN_states,num_timesteps,1)

        
        self.build(num_inputs)

        
    def build(self, inputs_shape):
        if inputs_shape is None:
            raise ValueError("Expected inputs.shape to be known, saw shape: %s"
                            % str(inputs_shape))

        self.input_depth = inputs_shape   
        h_depth = self._num_units
        self._kernel = self.add_variable(
            'kernel',
            shape=[self.input_depth + h_depth+1, 4 * self._num_units])
        self._bias = self.add_variable(
            'bias',
            shape=[4 * self._num_units],
            initializer=tf.zeros_initializer(dtype=tf.float32))

        #weights for input attention 
        self._We = self.add_variable(
            'We',
            shape=[self._num_units*2,self.timesteps])

        self._Bwe = self.add_variable(
            'Bwe',
            shape=[self.timesteps],
            initializer=tf.zeros_initializer(dtype=tf.float32)
            )   

        self._Ue = self.add_variable(
            '_Ue',
            shape=[self.timesteps,self.timesteps])

        self._Bue = self.add_variable(
            'Bue',
            shape=[self.timesteps],
            initializer=tf.zeros_initializer(dtype=tf.float32)
            )    

        self._v1 = self.add_variable(
            'v1',
            shape=[self.timesteps,1])
        ##########################
        #weights for temporal attention
        self._Wd = self.add_variable(
            'Wd',
            shape=[self._num_units*2,self.input_depth])           

        self._Ud = self.add_variable(
            '_Uwd',
            shape=[self.input_depth,self.input_depth])

        self._Bwd = self.add_variable(
            'Bwd',
            shape=[self.input_depth],
            initializer=tf.zeros_initializer(dtype=tf.float32)
            )   
        self._Bud = self.add_variable(
            'Bud',
            shape=[self.input_depth],
            initializer=tf.zeros_initializer(dtype=tf.float32)
            )    

        self._v2 = self.add_variable(
            'v2',
            shape=[self.input_depth,1])     

        self._Wy = self.add_variable(
            '_Wy',
            shape=[self.input_depth+1,1])

        self._By = self.add_variable(
            'By',
            shape=[1],
            initializer=tf.zeros_initializer(dtype=tf.float32)
            )  

        print(self._Wd.name)



        self.built = True
        
        
    def __call__(self,inputs,state,scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
        # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h, h2 = tf.split(state,3,1)     

            tempE = tf.matmul(tf.concat([h,h2],1),self._We)+self._Bwe            
            alphas=[]
            for inp in self.CNNinputs_for_IA:
                tempU = tf.matmul(tf.squeeze(inp),self._Ue)+self._Bue
                tempF = tf.tanh(tempE+tempU)
                
                alpha = tf.matmul(tempF,self._v1)
                alphas.append(alpha)
            
            alphas = tf.concat(alphas,1)
            alphas = tf.nn.softmax(alphas)
            alphas = tf.reshape(alphas,[-1,1,self.input_depth])
            Xinputs = self.inputs*alphas             


            tempE = tf.matmul(tf.concat([h,h2],1),self._Wd)+self._Bwd
            alphas=[]
            for inp in self.CNNinputs_for_TA:
                tempU = tf.matmul(tf.squeeze(inp),self._Ud)+self._Bud
                tempF = tf.tanh(tempE+tempU)
                
                alpha = tf.matmul(tempF,self._v2)
                alphas.append(alpha)
            alphas = tf.concat(alphas,1)
            alphas = tf.reshape(tf.nn.softmax(alphas),[-1,self.timesteps,1])

            context = tf.multiply(alphas,Xinputs)
            context = tf.squeeze(tf.reduce_sum(context,axis=1))    

            gate_inputs = tf.matmul(
            tf.concat([inputs,context, h], 1), self._kernel)
            gate_inputs = tf.add(gate_inputs, self._bias)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(gate_inputs, 4, 1)

            old_h = h
            new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
                    self._activation(j))
            new_h = self._activation(new_c) * tf.sigmoid(o)

            if self._state_is_tuple:
                new_state = rnn.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h, old_h],1)
        return new_h, new_state

"""
import numpy as np

inputs = tf.constant([1,2,3,4,5,6,7,8,9,10]*10,shape=[2,5,10],dtype=tf.float32)
softmax= tf.nn.softmax(tf.ones([2,10]))
softmax=tf.reshape(softmax,[-1,1,10])
#y= tf.ones([2,5,1],dtype=tf.float32)

#DAL = DualAttnLayer(16,10,5,inputs,y,2)
#result = DAL()

result=inputs*softmax

sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(inputs))
print(sess.run(softmax))
print(sess.run([result,tf.shape(result)]))
"""