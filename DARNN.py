import tensorflow as tf
rnn = tf.nn.rnn_cell

class InputAttention_LSTMCell(rnn.BasicLSTMCell):
    """
    MY LSTM proposed.
    """
    def __init__(self,
               num_units,
               num_inputs,
               num_timesteps,
               inputs,
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
        super(InputAttention_LSTMCell,self).__init__(num_units,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs)

        self.inputs=tf.split(inputs,num_inputs,2)
        self.timesteps=num_timesteps
        self.build(num_inputs)        
        
    def build(self, inputs_shape):
        if inputs_shape is None:
            raise ValueError("Expected inputs.shape to be known, saw shape: %s"
                            % str(inputs_shape))

        self.input_depth = inputs_shape   
        h_depth = self._num_units
        self._kernel = self.add_variable(
            'kernel',
            shape=[self.input_depth + h_depth, 4 * self._num_units])
        self._bias = self.add_variable(
            'bias',
            shape=[4 * self._num_units],
            initializer=tf.zeros_initializer(dtype=tf.float32))

        self._We = self.add_variable(
            'We',
            shape=[self._num_units*2,self.timesteps])

        self._Bw = self.add_variable(
            'Bw',
            shape=[self.timesteps],
            initializer=tf.zeros_initializer(dtype=tf.float32)
            )   

        self._Ue = self.add_variable(
            '_Ue',
            shape=[self.timesteps,self.timesteps])

        self._Bu = self.add_variable(
            'Bu',
            shape=[self.timesteps],
            initializer=tf.zeros_initializer(dtype=tf.float32)
            )    

        self._v = self.add_variable(
            'v',
            shape=[self.timesteps,1])
        

        self.built = True
        
    def __call__(self,inputs,state,scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
        # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(1, 2, state)

            tempE = tf.matmul(tf.concat([c,h],1),self._We)+self._Bw            
            
            alphas=[]
            for inp in self.inputs:
                tempU = tf.matmul(tf.squeeze(inp),self._Ue)+self._Bu
                tempF = tf.tanh(tempE+tempU)
                
                alpha = tf.matmul(tempF,self._v)
                alphas.append(alpha)
            
            alphas = tf.concat(alphas,1)
            alphas = tf.nn.softmax(alphas)
            inputs = tf.multiply(inputs,alphas)

            print(tf.shape(tempE),tf.shape(tempU),tf.shape(alpha),tf.shape(alphas),tf.shape(inputs))

            gate_inputs = tf.matmul(
            tf.concat([inputs, h], 1), self._kernel)
            gate_inputs = tf.add(gate_inputs, self._bias)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(gate_inputs, 4, 1)

            new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
                    self._activation(j))
            new_h = self._activation(new_c) * tf.sigmoid(o)

            if self._state_is_tuple:
                new_state = rnn.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat(1, [new_c, new_h])
        return new_h, new_state

class TemporalAttention_LSTMCell(rnn.BasicLSTMCell):
    """
    MY LSTM proposed.
    """
    def __init__(self,
               num_units,
               num_inputs,
               num_timesteps,
               encoderUnits,
               encoderStates,
               forget_bias=1.0,
               state_is_tuple=False,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs):
        """
        Initialize the basic LSTM cell.
        """        
        super(TemporalAttention_LSTMCell,self).__init__(num_units,
               forget_bias=1.0,
               state_is_tuple=False,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs)


        self.encoderRaw=encoderStates #shape = [batch, timesteps, encoderunits]
        self.encoderStates=tf.split(encoderStates,num_timesteps,1)
        self.timesteps=num_timesteps
        self.build(num_inputs,encoderUnits)        
        
    def build(self, inputs_shape,encoderUnits):
        if inputs_shape is None:
            raise ValueError("Expected inputs.shape to be known, saw shape: %s"
                            % str(inputs_shape))

        self.input_depth = inputs_shape   
        self.encoderUnits=encoderUnits
        h_depth = self._num_units
        self._kernel = self.add_variable(
            'kernel',
            shape=[self.input_depth + h_depth, 4 * self._num_units])
        self._bias = self.add_variable(
            'bias',
            shape=[4 * self._num_units],
            initializer=tf.zeros_initializer(dtype=tf.float32))

        self._Wd = self.add_variable(
            'Wd',
            shape=[self._num_units*2,self.encoderUnits])           

        self._Ud = self.add_variable(
            '_Ud',
            shape=[self.encoderUnits,self.encoderUnits])

        self._Bw = self.add_variable(
            'Bw',
            shape=[self.encoderUnits],
            initializer=tf.zeros_initializer(dtype=tf.float32)
            )   
        self._Bu = self.add_variable(
            'Bu',
            shape=[self.encoderUnits],
            initializer=tf.zeros_initializer(dtype=tf.float32)
            )    

        self._v = self.add_variable(
            'v',
            shape=[self.encoderUnits,1])     

        self._Wy = self.add_variable(
            '_Wy',
            shape=[self.encoderUnits+1,1])

        self._By = self.add_variable(
            'By',
            shape=[1],
            initializer=tf.zeros_initializer(dtype=tf.float32)
            )  

        self.built = True
        
    def __call__(self,inputs,state,scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
        # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h, cont = tf.split(state,3,1)

            tempE = tf.matmul(tf.concat([c,h],1),self._Wd)+self._Bw         
            
            alphas=[]
            for inp in self.encoderStates:
                tempU = tf.matmul(tf.squeeze(inp),self._Ud)+self._Bu
                tempF = tf.tanh(tempE+tempU)
                
                alpha = tf.matmul(tempF,self._v)
                alphas.append(alpha)
            
            alphas = tf.concat(alphas,1)
            alphas = tf.reshape(tf.nn.softmax(alphas),[-1,self.timesteps,1])

            context = tf.multiply(alphas,self.encoderRaw)
            context = tf.squeeze(tf.reduce_sum(context,axis=1))

            inputs = tf.matmul(tf.concat([inputs,context],1),self._Wy)+self._By
            
            print(tf.shape(tempE),tf.shape(tempU),tf.shape(alpha),tf.shape(alphas),tf.shape(inputs))

            gate_inputs = tf.matmul(
            tf.concat([inputs, h], 1), self._kernel)
            gate_inputs = tf.add(gate_inputs, self._bias)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(gate_inputs, 4, 1)

            new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
                    self._activation(j))
            new_h = self._activation(new_c) * tf.sigmoid(o)

            if self._state_is_tuple:
                new_state = rnn.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h, context],1)
                new_state = tf.reshape(new_state,[-1,192])

        return new_h, new_state

"""
encoderState = tf.ones([10,5,5],dtype = tf.float32)
inputs = tf.ones([10,5,1],dtype = tf.float32)

lstm = TemporalAttention_LSTMCell(5,1,5,5,encoderState)
initial_state=tf.zeros([10,15])
_,outputs = tf.nn.dynamic_rnn(lstm,inputs,initial_state=initial_state)


sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(outputs),sess.run(tf.shape(outputs)))
"""