from Models import *
import seaborn as sns
from memory_profiler import profile

"""
---------------------------------------------
UTILITY FUNCTIONS
---------------------------------------------

Functions that are used to carry out simple manipulations repeatedly.

"""

#Gives the paramaters of a normal distribution at 99.7% for a given range 
def compute_parameter_mu_sigma(range):
    mu = (range[0] + range[1])/2
    sigma = (range[1] - mu)/3
    return([mu, sigma])

#Gives a random number in a range with a normal distribution (mu, sigma)
def give_param(mu,sigma,range):
    param=np.random.normal(mu, sigma)
    while(param<range[0] or param>range[1]):
        param=np.random.normal(mu, sigma)
    return param 

"""
---------------------------------------------
CALCULATION FUNCTIONS 
---------------------------------------------

Functions that are used to perform calculations to get optimal clipping values
and optimal ranges for normal distribution. 

"""

#Computes the ideal clipping value on inputs with given clipping percentage on a specific activation layer  
def clipping_value_by_percentage(testing_model, clipping_percentage, activation_index, inputs, positive_values_only = False):
    
    layer_outputs = testing_model.get_activation_layer_outputs(inputs, activation_index, activation = True)

    if positive_values_only :
        clipping_value = np.percentile(layer_outputs[layer_outputs > 0], 100 - clipping_percentage)
    else :
        clipping_value = np.percentile(layer_outputs, 100 - clipping_percentage)
    return clipping_value


#Computes ideal clipping values on inputs with given clipping percentage for every activation layer
def clipping_list_by_percentage(testing_model, clipping_percentage, inputs):
    
    clipping_list = []
    c_relus_list = []
    model_copy = testing_model.clone()
    
    for i in range(model_copy.activation_nb):
        clipping_value = clipping_value_by_percentage(model_copy, clipping_percentage, i, inputs)
        clipping_list.append(clipping_value)
        new_c_relu = model_copy.c_relus_list[i].change_parameter('clipping', clipping_value)
        c_relus_list.append(new_c_relu)
        model_copy.change_c_relu(new_c_relu, i)

    del model_copy
    return clipping_list, c_relus_list

#For a given model and a parameter ['beta', 'gamma', 'clipping_percentage'], it gives the min and max parameters in accordance with the variation of precision and loss allowed
def compute_range(testing_model,parameter_name, allowed_accuracy_variation, allowed_loss_variation, significant_number_max, significant_number_min, dataset, inputs=None):
    
    model_copy = testing_model.clone()
    model_copy.change_all_c_relus(CustomReLU(1,0,np.inf))
    
    if not parameter_name in ['beta', 'gamma', 'clipping_percentage']:
        raise AssertionError("Parameter must be 'beta', 'gamma' or 'clipping_percentage'")
    
    parameter_max = np.inf
    parameter_min = -np.inf
    parameter = 0
    if(parameter_name == 'beta'):
        parameter = 1
    main_loss,main_accuracy = model_copy.evaluate(dataset)
    accuracy = np.inf
    loss = 0

    if(parameter_name == 'clipping_percentage'):
        parameter_min = 0
        while accuracy > main_accuracy - allowed_accuracy_variation and loss < main_loss + allowed_loss_variation :
            parameter_max = parameter
            print("Model testing with {} = {}".format(parameter_name,parameter))
            _, c_relus_list = clipping_list_by_percentage(model_copy, parameter, inputs)
            for i in range(model_copy.activation_nb):
                model_copy.change_c_relu(c_relus_list[i], i)
            
            loss, accuracy = model_copy.evaluate(dataset)
            parameter = np.round(parameter + pow(10,-significant_number_max),significant_number_max)
    else:
        while accuracy > main_accuracy - allowed_accuracy_variation and loss < main_loss + allowed_loss_variation :
            parameter_max = parameter
            print("Model testing with {} = {}".format(parameter_name,parameter))
            if(parameter_name == 'beta'):
                model_copy.change_all_c_relus(CustomReLU(parameter, 0, np.inf))
            else:
                model_copy.change_all_c_relus(CustomReLU(1, parameter, np.inf))
            loss, accuracy = model_copy.evaluate(dataset)
            parameter = np.round(parameter + pow(10,-significant_number_max), significant_number_max)
        
        parameter = 0
        if(parameter_name == 'beta'):
            parameter = 1
        accuracy = np.inf
        loss = 0

        while accuracy > main_accuracy - allowed_accuracy_variation and loss < main_loss + allowed_loss_variation :
            parameter_min = parameter
            print("Model testing with {} = {}".format(parameter_name,parameter))
            if(parameter_name == 'beta'):
                model_copy.change_all_c_relus(CustomReLU(parameter, 0, np.inf))
            else:
                model_copy.change_all_c_relus(CustomReLU(1, parameter, np.inf))
            loss, accuracy = model_copy.evaluate(dataset)
            parameter = np.round(parameter - pow(10,-significant_number_min),significant_number_min)
    del model_copy
    return([parameter_min,parameter_max])
        

"""
---------------------------------------------
PLOT FUNCTIONS 
---------------------------------------------

Functions that are used to plot the performance of the models 
according to their characteristics.

"""

#Plots accuracy of model on a given dataset with one parameter varying between min and max
    #Also plots loss if plot_loss = True
def plot_varying_parameter_accuracy(testing_model, parameter, parameter_min, parameter_max, parameter_step, dataset, plot_loss=False):
    
    X = np.arange(parameter_min, parameter_max, parameter_step)
    Y = []
    Y_loss = []
    model_copy = testing_model.clone()

    fig = plt.figure(constrained_layout=True)

    if not parameter in ['beta', 'gamma', 'clipping']:
        raise AssertionError("Parameter must be 'beta', 'gamma' or 'clipping'")
    else:
        for x in X :
            print("Testing model accuracy with {} = {}".format(parameter,x))
            if parameter == 'beta':
                model_copy.change_all_c_relus(CustomReLU(x, 0, np.inf))
            if parameter == 'gamma':
                model_copy.change_all_c_relus(CustomReLU(1, x, np.inf))
            if parameter == 'clipping': 
                model_copy.change_all_c_relus(CustomReLU(1, 0, x))
                
            loss, accuracy = model_copy.evaluate(dataset)
            Y.append(accuracy)
            Y_loss.append(loss)
        

        if plot_loss:
            plt.subplot(2,1,2)
            plt.plot(X,Y_loss, "o", label="Variation of {}".format(parameter))
            plt.title('Loss of model as a function of {} - Model : {}'.format(parameter, testing_model.name))
            plt.xlabel(parameter)
            plt.ylabel('loss')
            plt.legend()
            plt.subplot(2,1,1)
        
        
        plt.plot(X,Y, "o", label="Variation of {}".format(parameter))
        plt.title('Accuracy of model as a function of {} - Model : {}'.format(parameter, testing_model.name))
        plt.xlabel(parameter)
        plt.ylabel('accuracy')
        plt.ylim(0,1)
        plt.legend()
        del model_copy
        return fig


#Plots histograms of a specific activation layer on given inputs with one parameter varying between min and max
def plot_varying_parameter_hist(testing_model, parameter, parameter_min, parameter_max, parameter_step, inputs, activation_index=0, no_0=False):
    
    X = np.arange(parameter_min, parameter_max, parameter_step)
    figs = []

    model_copy = testing_model.clone()

    if not parameter in ['beta', 'gamma', 'clipping']:
        raise AssertionError("Parameter must be 'beta', 'gamma' or 'clipping'")
    else:
        for x in X :
            if parameter == 'beta':
                model_copy.change_all_c_relus(CustomReLU(x, 0, np.inf))
            if parameter == 'gamma':
                model_copy.change_all_c_relus(CustomReLU(1, x, np.inf))
            if parameter == 'clipping':
                model_copy.change_all_c_relus(CustomReLU(1, 0, x))

            outputs_before_activation = model_copy.get_activation_layer_outputs(inputs, activation_index, activation=False)
            outputs_after_activation = model_copy.get_activation_layer_outputs(inputs, activation_index, activation=True)

            fig = plt.figure()

            counts, bins = np.histogram(outputs_before_activation, bins=30)
            plt.hist(bins[:-1],bins, weights=counts, label="Before activation")

            if(no_0):
                counts, bins = np.histogram(delete_zeros(outputs_after_activation), bins=30)
                plt.hist(bins[:-1],bins, weights=counts, label='After activation')

            else:
                counts, bins = np.histogram(outputs_after_activation, bins=30)
                plt.hist(bins[:-1],bins , weights=counts, label='After activation')

            plt.title("Model distribution of outputs at layer {} with {} = {} - Model : {}".format(activation_index, parameter, x, testing_model.name))
            plt.xlabel("Outputs values")
            plt.ylabel("# Elements")
            plt.grid(True)
            plt.legend(loc="upper right")
            figs.append(fig)
        del model_copy
        return figs



#Plots accuracy of model on a given dataset with one parameter varying between min and max
    # inputs allows to calculate the clipping value according to the clipping percentage 
    # dataset allows to test the accuracy of the model after clipping
def plot_varying_clipping_percentage_accuracy(testing_model, clipping_percentage_min, clipping_percentage_max, clipping_percentage_step, dataset, inputs, plot_loss = False):
    if (clipping_percentage_min < 0 or clipping_percentage_max < 0 or clipping_percentage_min > 100 or clipping_percentage_max > 100):
        raise AssertionError("Invalid clipping percentage parameters")
    else : 
        model_copy = testing_model.clone()

        fig = plt.figure(constrained_layout=True)

        X_clipping_percentage = np.arange(clipping_percentage_min, clipping_percentage_max, clipping_percentage_step)
        Y_clipping_percentage = []
        Y_clipping_percentage_loss = []

        for x_clipping_percentage in X_clipping_percentage:
            _, c_relus_list = clipping_list_by_percentage(testing_model, x_clipping_percentage, inputs)
            for i in range(model_copy.activation_nb):
                model_copy.change_c_relu(c_relus_list[i], i)
            loss, accuracy = model_copy.evaluate(dataset)
            Y_clipping_percentage.append(accuracy)
            Y_clipping_percentage_loss.append(loss)

        if plot_loss:
            plt.subplot(2,1,2)
            plt.plot(X_clipping_percentage,Y_clipping_percentage_loss, "o", label="Variation of clipping percentage")
            plt.title('Loss of model as a function of clipping percentage - Model : {}'.format(testing_model.name))
            plt.xlabel('clipping percentage')
            plt.ylabel('loss')
            plt.legend()
            plt.subplot(2,1,1)

        plt.plot(X_clipping_percentage,Y_clipping_percentage, "o")
        plt.title("Accuracy of model as a function of clipping percentage - Model : {}".format(testing_model.name))
        plt.xlabel("clipping percentage")
        plt.ylabel("accuracy")
        plt.ylim(0,1)
        del model_copy
        return fig

#Plots accuracy of model on a given dataset with two parameter varying between min and max   
    # x and y are chosen according to the order of the parameters in argument 
def plot_heatmap_varying_parameters_accuracy(testing_model, parameter_list_x, parameter_list_y, dataset, inputs):
    #parameter_list format : [parameter, parameter_min, parameter_max, parameter_step]
    parameters_matrix = [parameter_list_x] + [parameter_list_y]
    parameters = [parameters_matrix[0][0], parameters_matrix[1][0]]

    if not all(parameter in ['beta', 'gamma', 'clipping', 'clipping_percentage'] for parameter in parameters) or parameters[0] == parameters[1] or all(parameter in ['clipping', 'clipping_percentage'] for parameter in parameters):
        raise AssertionError("Invalid parameters, parameters must be different from each other and be 'beta', 'gamma', 'clipping' or 'clipping_percentage'.")
    else :
        _, parameter_x_min, parameter_x_max, parameter_x_step = parameters_matrix[0]
        X_parameter = np.arange(parameter_x_min, parameter_x_max, parameter_x_step)

        _, parameter_y_min, parameter_y_max, parameter_y_step = parameters_matrix[1]
        Y_parameter = np.arange(parameter_y_min, parameter_y_max, parameter_y_step)

        Z = []

        model_copy = testing_model.clone()
        counter = 0
        for x in X_parameter:
            Z.append([])
            for y in Y_parameter:
                print("Model testing with {} = {} and {} = {}".format(parameters[0], x, parameters[1], y))

                if 'clipping_percentage' in parameters :
                    index = parameters.index('clipping_percentage')

                    c_relu = CustomReLU(1,0,np.inf)
                    if index-1 == 0:
                        new_c_relu = c_relu.change_parameter(parameters[index-1], x)
                        model_copy.change_all_c_relus(new_c_relu)
                        clipping_list, _ = clipping_list_by_percentage(model_copy, y, inputs)
                    else :
                        new_c_relu = c_relu.change_parameter(parameters[index-1], y)
                        model_copy.change_all_c_relus(new_c_relu)
                        clipping_list, _ = clipping_list_by_percentage(model_copy, x, inputs)

                    for i in range(model_copy.activation_nb) :
                        new_c_relu = new_c_relu.change_parameter('clipping', clipping_list[i])
                        model_copy.change_c_relu(new_c_relu, i)
                        
                else : 
                    c_relu = CustomReLU(1,0,np.inf)
                    new_c_relu = c_relu.change_parameter(parameters[0], x)
                    new_c_relu  = new_c_relu.change_parameter(parameters[1], y)

                    model_copy.change_all_c_relus(new_c_relu)
                    
                _, accuracy = model_copy.evaluate(dataset)
                Z[counter].append(accuracy)
            counter += 1
        
        p = sns.heatmap(Z ,annot=True, annot_kws={"size": 6}, xticklabels = np.round(Y_parameter, 2), yticklabels = np.round(X_parameter, 2))
        p.set_xlabel(parameters[1])
        p.set_ylabel(parameters[0])
        plt.title("Accuracy of model as a function of {} and {}".format(parameters[0], parameters[1]))
        fig = p.get_figure()
        del model_copy
        return fig

"""
---------------------------------------------
PLOT FUNCTIONS for Hamming Weight
---------------------------------------------
"""
    
#Plots a 3D histogram of the hamming weight distribution against several models on a given layer
def plot_varying_model_HW_dist(testing_model_list, inputs, activation_index=0, diff=False):
    HW_layer_output_hist = []
    param_hist =[]
    number_model = -1
    fig=plt.figure(figsize=[9.6,9.6])
    for testing_model in testing_model_list :
        number_model+=1

        outputs_after_activation = testing_model.get_activation_layer_outputs(inputs, activation_index, activation=True)
        HW_layer_output=to_HW_layer_output(outputs_after_activation)

        for HW in HW_layer_output:
            HW_layer_output_hist.append(HW)  
        param_hist+=len(HW_layer_output)*[number_model]
        

    x, y = HW_layer_output_hist,param_hist
    x=np.array(x)

    
    hist, xedges, yedges = np.histogram2d(x, y, bins=[32,len(testing_model_list)],range=[[0,32],[0, len(testing_model_list)]])
    

    # Construct arrays for the anchor positions of the bars.
    xpos, ypos = np.meshgrid(xedges[:-1] , yedges[:-1] , indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    dx = 1 * np.ones_like(zpos)
    dy = 0.5 * np.ones_like(zpos)
    
    # Construct arrays with the dimensions for the bars.
    if(diff):
        hist_diff=copy.deepcopy(hist)
        ax2 = fig.add_subplot(212,projection='3d')
        ref_outputs_after_activation = testing_model.get_activation_layer_outputs(inputs, activation_index, activation=True)
        ref_HW_layer_output=to_HW_layer_output(ref_outputs_after_activation)
        nb_bit=np.array([x for x in range(32)])
        counts, bins = np.histogram(ref_HW_layer_output, bins=nb_bit)
        for i in range(len(counts)):
                hist_diff[i]=abs(hist_diff[i]-counts[i])
        dz_diff = hist_diff.ravel()
        ax2.bar3d(xpos, ypos, zpos, dx, dy, dz_diff, zsort='average')
        plt.title("HW distribution diff of different model with Custom ReLU - layer {} - Model : {}".format(activation_index, testing_model_list[0].name))
        plt.xlabel("hamming weight")
        plt.ylabel("model")

    dz = hist.ravel()
      
    ax1 = fig.add_subplot(211,projection='3d')
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

    plt.title("HW distribution of different model with Custom ReLU")
    plt.xlabel("hamming weight")
    plt.ylabel("model")
    return fig


#Plots a 3D histogram of the hamming weight distribution versus the variation of a parameter ['beta', 'gamma', 'clipping'] 
    #for a given model
def plot_varying_parameter_HW_dist(testing_model, parameter, parameter_min, parameter_max, parameter_step, inputs,
                                   activation_index=0,diff=False,no_0=False):
    PARAM = np.arange(parameter_min, parameter_max+parameter_step, parameter_step)
    HW_layer_output_hist = []
    param_hist =[]
    model_copy = testing_model.clone()
    fig=plt.figure(figsize=[9.6,9.6])
    if not parameter in ['beta', 'gamma', 'clipping']:
        raise AssertionError("Parameter must be 'beta', 'gamma' or 'clipping'")
    else:
        for param in PARAM :
            if parameter == 'beta':
                model_copy.change_all_c_relus(CustomReLU(param, 0, np.inf))
            if parameter == 'gamma':
                model_copy.change_all_c_relus(CustomReLU(1, param, np.inf))
            if parameter == 'clipping':
                model_copy.change_all_c_relus(CustomReLU(1, 0, param))



            outputs_after_activation = model_copy.get_activation_layer_outputs(inputs, activation_index, activation=True)
            HW_layer_output=to_HW_layer_output(outputs_after_activation)
            for HW in HW_layer_output:
                HW_layer_output_hist.append(HW)  
            param_hist+=len(HW_layer_output)*[param]
        
        x, y = HW_layer_output_hist,param_hist
        x=np.array(x)

        if no_0:
            hist, xedges, yedges = np.histogram2d(x, y, bins=[31,len(PARAM)],range=[[1,32],[parameter_min, parameter_max+parameter_step]])
        else :
            hist, xedges, yedges = np.histogram2d(x, y, bins=[32,len(PARAM)],range=[[0,32],[parameter_min, parameter_max+parameter_step]])
        
    
        # Construct arrays for the anchor positions of the bars.
        xpos, ypos = np.meshgrid(xedges[:-1] , yedges[:-1] , indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        # Construct arrays with the dimensions for the bars.
       
        dx =   np.ones_like(zpos)
        dy = parameter_step/1.5 * np.ones_like(zpos)
        

        if(diff):
            hist_diff=copy.deepcopy(hist)
            ax2 = fig.add_subplot(212,projection='3d')
            ref_outputs_after_activation = testing_model.get_activation_layer_outputs(inputs, activation_index, activation = True)
            ref_HW_layer_output=to_HW_layer_output(ref_outputs_after_activation)
            nb_bit=np.array([x for x in range(32)])
            counts, bins = np.histogram(ref_HW_layer_output, bins=nb_bit)
            for i in range(len(counts)):
                 hist_diff[i]=abs(hist_diff[i]-counts[i])
            dz_diff = hist_diff.ravel()
            ax2.bar3d(xpos, ypos, zpos, dx, dy, dz_diff, zsort='average')
            plt.title("HW distribution diff with {} variation for Model : {}".format(parameter,testing_model.name))
            plt.xlabel("hamming weight")
            plt.ylabel("{}".format(parameter))

        dz = hist.ravel()
      
        ax1 = fig.add_subplot(211,projection='3d')
        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

        plt.title("HW distribution with {} variation for Model : {}".format(parameter,testing_model.name))
        plt.xlabel("hamming weight")
        plt.ylabel("{}".format(parameter))
        del model_copy
        return fig


#Plots a 3D histogram of the difference in hamming weights at the output of the neuron with respect to the variation 
    #of a parameter ['beta', 'gamma', 'clipping']  for a given model
def plot_varying_parameter_HW_diff(testing_model, parameter, parameter_min, parameter_max, parameter_step, inputs,
                                   activation_index = 0,no_0=False):
    
    PARAM = np.arange(parameter_min, parameter_max+parameter_step, parameter_step)
    HW_layer_output_hist = []
    param_hist =[]
    
    if not parameter in ['beta', 'gamma', 'clipping']:
        raise AssertionError("Parameter must be 'beta', 'gamma' or 'clipping'")
    else:

        model_copy = testing_model.clone()
        ref_outputs_after_activation = testing_model.get_activation_layer_outputs(inputs, activation_index, activation=True)
        ref_HW_layer_output=to_HW_layer_output(ref_outputs_after_activation)

        
        for param in PARAM :
            if parameter == 'beta':
                model_copy.change_all_c_relus(CustomReLU(param, 0, np.inf))
            if parameter == 'gamma':
                model_copy.change_all_c_relus(CustomReLU(1, param, np.inf))
            if parameter == 'clipping':
                model_copy.change_all_c_relus(CustomReLU(1, 0, param))


            outputs_after_activation = model_copy.get_activation_layer_outputs(inputs, activation_index, activation=True)
            HW_layer_output=to_HW_layer_output(outputs_after_activation)

            nb_diff=0
            for i in range(len(HW_layer_output)):
                
                if(HW_layer_output[i]!=ref_HW_layer_output[i]):
                    HW_layer_output_hist.append(HW_layer_output[i]) 
                    nb_diff+=1
        
            param_hist+=nb_diff*[param]
        
    
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        x, y = HW_layer_output_hist,param_hist
        x=np.array(x)
        
        if no_0:
            hist, xedges, yedges = np.histogram2d(x, y, bins=[31,len(PARAM)],range=[[1,32],[parameter_min, parameter_max+parameter_step]])
        else :
            hist, xedges, yedges = np.histogram2d(x, y, bins=[32,len(PARAM)],range=[[0,32],[parameter_min, parameter_max+parameter_step]])
        
        # Construct arrays for the anchor positions of the bars.
        xpos, ypos = np.meshgrid(xedges[:-1] , yedges[:-1] , indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        # Construct arrays with the dimensions for the bars.
        dx =   np.ones_like(zpos)
        dy = parameter_step/1.5 * np.ones_like(zpos)
        dz = hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

        plt.title("HW diff with {} variation for Model : {} at layer {}".format(parameter,testing_model.name,activation_index))
        plt.xlabel("hamming weight")
        plt.ylabel("{}".format(parameter))
        del model_copy
        return fig
    

"""
---------------------------------------------
RANDOM MODEL MODIFICATION FUNCTIONS
---------------------------------------------

Functions that make random changes to the Custom ReLUs of a model.

"""

#Gives random parameters beta gamma and clipping depending of ranges and change the ReLU of the model with those parameters
def update_model_c_relus_with_ranges(testing_model, beta_range, gamma_range, clipping_percentage_range, inputs):
    beta_mu, beta_sigma = compute_parameter_mu_sigma(beta_range)
    random_beta = give_param(beta_mu, beta_sigma,beta_range)

    gamma_mu, gamma_sigma = compute_parameter_mu_sigma(gamma_range)
    random_gamma = give_param(gamma_mu, gamma_sigma,gamma_range)

    clipping_percentage_mu, clipping_percentage_sigma = compute_parameter_mu_sigma(clipping_percentage_range)
    random_percentage = give_param(clipping_percentage_mu, clipping_percentage_sigma,clipping_percentage_range)

    c_relu = CustomReLU(random_beta, random_gamma, np.inf)
    testing_model.change_all_c_relus(c_relu)
    clipping_list, _ = clipping_list_by_percentage(testing_model, random_percentage, inputs)
    
    for i in range(testing_model.activation_nb):
        new_c_relu = c_relu.change_parameter('clipping', clipping_list[i])
        testing_model.change_c_relu(new_c_relu, i)
    
    return random_beta, random_gamma, random_percentage

#Tests 10 models with a custom ReLU and plot their accuracy
def test_model_c_relus_with_ranges(testing_model, beta_range, gamma_range, clipping_percentage_range, dataset, inputs = None):
    model_copy = testing_model.clone()

    model_copy.change_all_c_relus(CustomReLU(1,0,np.inf))
    _, ref_accuracy = model_copy.evaluate(dataset)
    Y = []

    fig = plt.figure()
    for i in range(10):
        beta, gamma, clipping_percentage = update_model_c_relus_with_ranges(model_copy, beta_range, gamma_range, clipping_percentage_range, inputs)
        print("Testing model accuracy with beta = {}, gamma = {}, clipping percentage = {}".format(beta, gamma, clipping_percentage))
        _, accuracy = model_copy.evaluate(dataset)
        Y.append(accuracy)

    plt.plot(range(10), Y, "o")
    plt.title("Model accuracy with random parameters in ranges - Model : {}".format(model_copy.name))
    plt.xlabel("model iteration")
    plt.ylabel("accuracy")
    plt.hlines(ref_accuracy, xmin = 0, xmax = 10, linestyles='dashed', label = "Reference accuracy")
    plt.legend()
    
    return fig

#Allows you to test the accuracy of a model by changing the readback at each inference by a custom readback according to the given ranges.
    #The nb_test allows you to modify the number of times you test the whole dataset
#inputs is the X dataset and outputs is the Y dataset 
def update_model_c_relus_changing_every_inference(testing_model, beta_range, gamma_range, clipping_percentage_range, inputs, outputs, dataset,
                                                  nb_test = 10):
    model_copy = testing_model.clone()

    _, ref_accuracy = model_copy.evaluate(dataset)
    beta_mu, beta_sigma = compute_parameter_mu_sigma(beta_range)
    
    gamma_mu, gamma_sigma = compute_parameter_mu_sigma(gamma_range)

    clipping_percentage_mu, clipping_percentage_sigma = compute_parameter_mu_sigma(clipping_percentage_range)

    X_dataset = inputs
    List_ACCURACY=[]

    for k in range(nb_test):
        accuracy = 0
        for i in range(len(X_dataset)):
            random_beta = give_param(beta_mu, beta_sigma,beta_range)
            random_gamma = give_param(gamma_mu, gamma_sigma,gamma_range)
            random_percentage = give_param(clipping_percentage_mu, clipping_percentage_sigma,clipping_percentage_range)
            c_relu = CustomReLU(random_beta, random_gamma, np.inf)
            testing_model.change_all_c_relus(c_relu)
            clipping_list, _ = clipping_list_by_percentage(testing_model, random_percentage, inputs)
            
            for j in range(testing_model.activation_nb):
                new_c_relu = c_relu.change_parameter('clipping', clipping_list[j])
                testing_model.change_c_relu(new_c_relu, j)
            X_data = X_dataset[i][np.newaxis,:]
            predict = np.max(testing_model.predict(X_data))
            
            if(np.max(outputs[i])==predict):
                accuracy+=1
       
       
        List_ACCURACY+=[accuracy/len(X_dataset)]
        
    fig = plt.figure()
    plt.plot(range(nb_test), List_ACCURACY, "o")
    plt.title("Model accuracy with random parameters changing every inference - Model : {}".format(testing_model.name))
    plt.xlabel("model iteration")
    plt.ylabel("accuracy")
    plt.hlines(ref_accuracy, xmin = 0, xmax = nb_test, linestyles='dashed', label = "Reference accuracy")
    plt.legend()

    del model_copy
    return fig



"""
---------------------------------------------
PLOT FUNCTIONS VARYING SIGMA
---------------------------------------------
"""

#Plots accuracy of model on a given dataset with one parameter varying according to a normal distribution
def plot_varying_sigma_parameter_accuracy(testing_model, parameter, sigma_min, sigma_max, sigma_step, mu, dataset):
    
    sigma_tab = np.arange(sigma_min, sigma_max, sigma_step)
    Y = []

    model_copy = testing_model.clone()

    fig = plt.figure()

    if not parameter in ['beta', 'gamma', 'clipping']:
        raise AssertionError("Parameter must be 'beta', 'gamma' or 'clipping'")
    else:
        for sigma in sigma_tab :
            mean_accuracy = 0
            if parameter == 'beta':
                X = abs(np.random.normal(mu,sigma,10))
                for x in X :
                    print("Testing model accuracy with sigma = {} and {} = {}".format(sigma,parameter,x))
                    model_copy.change_all_c_relus(CustomReLU(x, 0, np.inf))
                    _, accuracy = model_copy.evaluate(dataset)
                    mean_accuracy+=accuracy
            if parameter == 'gamma':
                X = np.random.normal(mu,sigma,10)
                for x in X :
                    print("Testing model accuracy with sigma = {} and {} = {}".format(sigma,parameter,x))
                    model_copy.change_all_c_relus(CustomReLU(1, x, np.inf))
                    _, accuracy = model_copy.evaluate(dataset)
                    mean_accuracy+=accuracy
            if parameter == 'clipping': 
                X = np.random.normal(mu,sigma,10)
                for x in X :
                    print("Testing model accuracy with sigma = {} and {} = {}".format(sigma,parameter,x))
                    model_copy.change_all_c_relus(CustomReLU(1, 0, x))
                    _, accuracy = model_copy.evaluate(dataset)
                    mean_accuracy+=accuracy
                
            Y.append(mean_accuracy/len(X))

        plt.plot(sigma_tab,Y, "o", label="Variation of {}".format(parameter))
        plt.title('Accuracy of model as a function of sigma of {} - Model : {}'.format(parameter, testing_model.name))
        plt.xlabel('sigma')
        plt.ylabel('accuracy')
        plt.ylim(0,1)
        plt.legend()
        del model_copy
        return fig
    
def plot_varying_sigma_clipping_percentage_accuracy(testing_model, sigma_min, sigma_max, sigma_step,mu, dataset): 
    model_copy = testing_model.clone()

    fig = plt.figure()

    sigma_tab = np.arange(sigma_min, sigma_max, sigma_step)
    Y_clipping_percentage = []
    
    for sigma in sigma_tab :
        mean_accuracy = 0
        X_clipping_percentage = abs(np.random.normal(mu,sigma,10))

        for x_clipping_percentage in X_clipping_percentage:
            clipping_list, c_relus_list = clipping_list_by_percentage(testing_model, x_clipping_percentage)
            print(clipping_list)
            for i in range(model_copy.activation_nb):
                model_copy.change_c_relu(c_relus_list[i], i)
            _, accuracy = model_copy.evaluate(dataset)
            mean_accuracy += accuracy
        Y_clipping_percentage.append(mean_accuracy/len(X_clipping_percentage))

    plt.plot(sigma_tab,Y_clipping_percentage, "o")
    plt.title("Accuracy of model as a function of sigma of clipping percentage - Model : {}".format(testing_model.name))
    plt.xlabel("sigma")
    plt.ylabel("accuracy")
    plt.ylim(0,1)
    del model_copy
    return fig