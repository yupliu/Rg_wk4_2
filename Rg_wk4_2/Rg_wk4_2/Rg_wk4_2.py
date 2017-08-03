import graphlab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sales = graphlab.SFrame('C:\\Machine_Learning\\Rg_wk4_2\\kc_house_data.gl')


#return 1+h0(xi)+h1(xi)+...., and output
def get_numpy_data(data_sframe,features,output):
    data_sframe['constant'] = 1
    features = ['constant'] + features
    features_sframe = data_sframe[features]
    features_matrix = features_sframe.to_numpy()
    output_sarray = data_sframe[output]
    output_array = output_sarray.to_numpy()
    return (features_matrix,output_array)


#(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') # the [] around 'sqft_living' makes it a list
#print example_features[0,:] # this accesses the first row of the data the ':' indicates 'all columns'
#print example_output[0] # and the corresponding output

#my_weights = np.ones(2)
#my_features = example_features[0,:]
#predicted_value = np.dot(my_weights,my_features)

#calculate the dot product of w*h
def predict_output(feature_matrix,weights):
    pred = np.dot(feature_matrix,weights)
    return pred


def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    if (feature_is_constant):
        derivative = 2*np.dot(errors,feature)
    else:
        derivative = 2*np.dot(errors,feature) + 2*l2_penalty*weight
    return derivative

#test derivative_ridge function
(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')
my_weights = np.array([1., 10.])
test_predictions = predict_output(example_features, my_weights)
errors = test_predictions - example_output # prediction errors

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:,1], my_weights[1], 1, False)
print np.sum(errors*example_features[:,1])*2+20.
print ''

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:,0], my_weights[0], 1, True)
print np.sum(errors)*2.

def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    #while not reached maximum number of iterations:
    # compute the predictions using your predict_output() function
    print 'Starting gradient descent with l2_penalty = ' + str(l2_penalty)
    weights = np.array(initial_weights)
    iteration = 0
    print_frequency = 1  # for adjusting frequency of debugging output
    gradient_sum_square = 0  # initialize the gradient sum of squares
    converged = False
    while not converged:        
        iteration += 1  # increment iteration counter
        ### === code section for adjusting frequency of debugging output. ===
        if iteration == 10:
            print_frequency = 10
        if iteration == 100:
            print_frequency = 100
        if iteration%print_frequency==0:
            print('Iteration = ' + str(iteration))
        # compute the errors as predictions - output
        prediction = predict_output(feature_matrix,weights)    
        errors = prediction - output
        # from time to time, print the value of the cost function
        if iteration%print_frequency==0:
            print 'Cost function = ', str(np.dot(errors,errors) + l2_penalty*(np.dot(weights,weights) - weights[0]**2))        
        for i in xrange(len(weights)): # loop over each weight
        # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
        # compute the derivative for weight[i].
        #(Remember: when i=0, you are computing the derivative of the constant!)
            if(i==0):
                derivative = feature_derivative_ridge(errors,feature_matrix[:,i],weights[i],l2_penalty,True)
            else:
                derivative = feature_derivative_ridge(errors,feature_matrix[:,i],weights[i],l2_penalty,False)
            
            #weights[i] = (1-2*step_size*l2_penalty)*weights[i] - step_size * derivative
            weights = weights-step_size*derivative
            #print weights[i]
            # compute the square-root of the gradient sum of squares to get the gradient magnitude:            
            if iteration > max_iterations:
                converged = True
    print 'Done with gradient descent at iteration ', iteration
    print 'Learned weights = ', str(weights)
    return weights

simple_features = ['sqft_living']
my_output = 'price'
train_data,test_data = sales.random_split(.8,seed=0)
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)
initial_weights = np.array([0., 0.])
step_size = 1e-12
max_iterations=1000
l2_penalty = 0.0
simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix,output,initial_weights,step_size,l2_penalty,max_iterations)
l2_penalty = 1e11
initial_weights = np.array([0., 0.])
simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix,output,initial_weights,step_size,l2_penalty,max_iterations)
plt.plot(simple_feature_matrix,output,'k.',
         simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_0_penalty),'b-',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_high_penalty),'r-')
plt.show()
