import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #### MARK: CODE WRITTEN, NOT TESTED
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : (1 / (1 + np.exp(-x)))  # Replace 0 with your sigmoid calculation. 
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        
        # MARK: just added these prints to see what the features look like..
#         print("FEATURES: \n", features)
#         print("TARGETS: \n", targets)
        
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        # MARK: CODE WRITTEN, NOT TESTED
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        
        # MARK: should use the sigmoid function to normalize ethe data here, these are now the features for the next
        # MARK: set of weights
        # MARK: CODE WRITTEN, NOT TESTED
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        # MARK: CODE WRITTEN, NOT TESTED
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = self.activation_function(final_inputs) # signals from final output layer
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###
        
        # TODO: Output error - Replace this value with your calculations.
        # MARK: I think this is (y - output)
        # MARK: CODE WRITTEN, NOT TESTED YET
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
#         print("FINAL OUTPUT: \n\n", final_outputs)
#         print("WHAT WE WANTED\n\n", y)
        print("ERROR\n\n", error)
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        # ERROR TERM FORMULA: δo =(y− y^)f′(W⋅a)
        # a: Sigmoid Activation function
        # sigmoid function derivative = sigmoid`(x) = sigmoid`(x) * (1 - sigmoid`(x))
        # MARK: CODE WRITTEN, NOT TESTED YET
#         print("ERROR", error, "\n\n")
#         print("FINAL OUTPUTS", final_outputs, "\n\n")
#         print("DESIRED VALUE", y, "\n\n")
        derivative_weights_final_output = (final_outputs * (1 - final_outputs))
#         print("DERIVATIVE WEIGHTS FINAL OUTPUT", derivative_weights_final_output, "\n\n\n") 
         
        output_error_term = error * derivative_weights_final_output
        
#         print("OUTPUT ERROR TERM:\n", output_error_term)
#         print("WEIGHTS HIDDEN TO OUTPUT:\n", self.weights_hidden_to_output)
        # TODO: Calculate the hidden layer's contribution to the error
        # MARK: CODE WRITTEN, NOT TESTED YET
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
        
#         print("HIDDEN ERROR\n\n", hidden_error)
        
#         # TODO: Backpropagated error terms - Replace these values with your calculations.
#         # MARK: CODE WRITTEN, NOT TESTED YET
#         output_error_term = error * output * (1 - output)
        # MARK: CODE WRITTEN, NOT TESTED YET
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        # Weight step (input to hidden)
        # MARK: CODE WRITTEN, NOT TESTED YET
        delta_weights_i_h += hidden_error_term * X[:, None]
        # Weight step (hidden to output)
        # MARK: CODE WRITTEN, NOT TESTED YET
        
#         print("OUTPUT_ERROR_TERM: \n\n", output_error_term)
#         print("HIDDEN OUTPUTS: \n\n", hidden_outputs)
        delta_weights_h_o_step = output_error_term * hidden_outputs[:,None]
#         print("STEP: \n\n", delta_weights_h_o_step)
#         print("DELTA H_O \n\n", delta_weights_h_o)
        delta_weights_h_o += delta_weights_h_o_step 
        print("HIDDEN ERROR TERM \n\n", hidden_error_term)
        print("X[:, None]\n\n", X[:, None])
        print("DELTA_WEIGHTS_INPUT_HIDDEN\n\n", delta_weights_i_h) 
        print("DELTA_WEIGHTS_HIDDEN_OUTPUT: \n\n", delta_weights_h_o)
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        
        # MARK: CODE WRITTEN, NOT TESTED YET
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records  # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        # MARK: Why is this correct? And why don't we run the activation function on it?
        final_outputs = final_inputs # signals from final output layer 
#         print("FINAL INPUTS: \n\n", final_inputs)
#         print("HIT THIS FINAL OUTPUT: \n\n", final_outputs)
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 100
learning_rate = 0.1
hidden_nodes = 2
# iterations = 1000
# learning_rate = 0.2
# hidden_nodes = 7
output_nodes = 1
