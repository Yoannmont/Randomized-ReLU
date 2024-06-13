
"""
---------------------------------------------
TESTS CONFIGURATION FILE FOR RELU PARAMETERS
---------------------------------------------
"""


"""
---------------------------------------------
CONFIG BETA
---------------------------------------------
"""


beta_config_MLP_MNIST={
    'min' : 0,
    'max' : 5,
    'step' : 0.05,
}

beta_config_CNN_MNIST={
    'min' : 0,
    'max' : 5,
    'step' : 0.05,
}

beta_config_CNN_CIFAR10={
    'min' : 0,
    'max' : 5,
    'step' : 0.05,
}


"""
---------------------------------------------
CONFIG GAMMA
---------------------------------------------
"""

gamma_config_MLP_MNIST={
    'min' : -5,
    'max' : 5,
    'step' : 0.05,
}

gamma_config_CNN_MNIST={
    'min' : -2,
    'max' : 2,
    'step' : 0.05,
}

gamma_config_CNN_CIFAR10={
    'min' : -0.75,
    'max' : 0.75,
    'step' : 0.01,
}


"""
---------------------------------------------
CONFIG CLIPPING
---------------------------------------------
"""

clipping_config_MLP_MNIST={
    'min' : 0,
    'max' : 20,
    'step' : 0.5,
}

clipping_config_CNN_MNIST={
    'min' : 0,
    'max' : 1,
    'step' : 0.01,
}

clipping_config_CNN_CIFAR10={
    'min' : 0,
    'max' : 1,
    'step' : 0.01,
}


"""
---------------------------------------------
CONFIG CLIPPING PERCENTAGE
---------------------------------------------
"""

clipping_percentage_config_MLP_MNIST={
    'min' : 0,
    'max' : 60,
    'step' : 1,
}

clipping_percentage_config_CNN_MNIST={
    'min' : 0,
    'max' : 30,
    'step' : 1,
}

clipping_percentage_config_CNN_CIFAR10={
    'min' : 0,
    'max' : 30,
    'step' : 1,
}



"""
---------------------------------------------
CONFIG BETA/GAMMA
---------------------------------------------
"""


beta_gamma_config_MLP_MNIST={
    'beta_min' : 0,
    'beta_max' : 2,
    'beta_step' : 0.1,
    'gamma_min':-3,
    'gamma_max':3,
    'gamma_step':0.5,

}

beta_gamma_config_CNN_MNIST={
    'beta_min' : 0,
    'beta_max' : 2,
    'beta_step' : 0.1,
    'gamma_min':-1,
    'gamma_max':1,
    'gamma_step':0.1,
}

beta_gamma_config_CNN_CIFAR10={
    'beta_min' : 0,
    'beta_max' : 2,
    'beta_step' : 0.1,
    'gamma_min':-0.4,
    'gamma_max':0.4,
    'gamma_step':0.04,
}



"""
---------------------------------------------
CONFIG BETA/CLIPPING
---------------------------------------------
"""


beta_clipping_config_MLP_MNIST={
    'beta_min' : 0,
    'beta_max' : 2,
    'beta_step' : 0.1,
    'clipping_min':0,
    'clipping_max':10,
    'clipping_step':0.5,

}

beta_clipping_config_CNN_MNIST={
    'beta_min' : 0,
    'beta_max' : 2,
    'beta_step' : 0.1,
    'clipping_min':0,
    'clipping_max':0.2,
    'clipping_step':0.01,
}

beta_clipping_config_CNN_CIFAR10={
    'beta_min' : 0,
    'beta_max' : 2,
    'beta_step' : 0.1,
    'clipping_min':0,
    'clipping_max':3,
    'clipping_step':0.1,
}



"""
---------------------------------------------
CONFIG BETA/CLIPPING PERCENTAGE
---------------------------------------------
"""


beta_clipping_percentage_config_MLP_MNIST={
    'beta_min' : 0,
    'beta_max' : 2,
    'beta_step' : 0.1,
    'clipping_percentage_min':0,
    'clipping_percentage_max':40,
    'clipping_percentage_step':2,

}

beta_clipping_percentage_config_CNN_MNIST={
    'beta_min' : 0,
    'beta_max' : 2,
    'beta_step' : 0.1,
    'clipping_percentage_min':0,
    'clipping_percentage_max':40,
    'clipping_percentage_step':2,
}

beta_clipping_percentage_config_CNN_CIFAR10={
    'beta_min' : 0,
    'beta_max' : 2,
    'beta_step' : 0.1,
    'clipping_percentage_min':0,
    'clipping_percentage_max':10,
    'clipping_percentage_step':0.5,
}


"""
---------------------------------------------
CONFIG GAMMA/CLIPPING
---------------------------------------------
"""


gamma_clipping_config_MLP_MNIST={
    'gamma_min' : -3,
    'gamma_max' : 3,
    'gamma_step' : 0.5,
    'clipping_min':0,
    'clipping_max':10,
    'clipping_step':0.5,

}

gamma_clipping_config_CNN_MNIST={
    'gamma_min' : -1,
    'gamma_max' : 1,
    'gamma_step' : 0.1,
    'clipping_min':0,
    'clipping_max':3,
    'clipping_step':0.1,
}

gamma_clipping_config_CNN_CIFAR10={
    'gamma_min' : -0.4,
    'gamma_max' : 0.4,
    'gamma_step' : 0.04,
    'clipping_min':0,
    'clipping_max':3,
    'clipping_step':0.1,
}



"""
---------------------------------------------
CONFIG GAMMA/CLIPPING PERCENTAGE
---------------------------------------------
"""


gamma_clipping_percentage_config_MLP_MNIST={
    'gamma_min' : -3,
    'gamma_max' : 3,
    'gamma_step' : 0.5,
    'clipping_percentage_min':0,
    'clipping_percentage_max':60,
    'clipping_percentage_step':2,

}

gamma_clipping_percentage_config_CNN_MNIST={
    'gamma_min' : -1,
    'gamma_max' : 1,
    'gamma_step' : 0.1,
    'clipping_percentage_min':0,
    'clipping_percentage_max':30,
    'clipping_percentage_step':1,
}

gamma_clipping_percentage_config_CNN_CIFAR10={
    'gamma_min' : -0.4,
    'gamma_max' : 0.4,
    'gamma_step' : 0.04,
    'clipping_percentage_min':0,
    'clipping_percentage_max':30,
    'clipping_percentage_step':1,
}



"""
---------------------------------------------
CONFIG MODELS
---------------------------------------------
"""

Config_model_list = []
    
config_CNN_CIFAR10={
    'name':'CIFAR10_model_CNN',
    'beta_config':beta_config_CNN_CIFAR10,
    'gamma_config':gamma_config_CNN_CIFAR10,
    'clipping_percentage_config':clipping_percentage_config_CNN_CIFAR10,
    'clipping_config':clipping_config_CNN_CIFAR10,
    
    'beta_clipping_percentage_config':beta_clipping_percentage_config_CNN_CIFAR10,
    'beta_clipping_config':beta_clipping_config_CNN_CIFAR10,
    'gamma_clipping_percentage_config':gamma_clipping_percentage_config_CNN_CIFAR10,
    'gamma_clipping_config':gamma_clipping_config_CNN_CIFAR10,
    'beta_gamma_config':beta_gamma_config_CNN_CIFAR10,
}
Config_model_list.append(config_CNN_CIFAR10)


config_CNN_MNIST={
    'name':'MNIST_model_CNN',
    'beta_config':beta_config_CNN_MNIST,
    'gamma_config':gamma_config_CNN_MNIST,
    'clipping_percentage_config':clipping_percentage_config_CNN_MNIST,
    'clipping_config':clipping_config_CNN_MNIST,
    
    'beta_clipping_percentage_config':beta_clipping_percentage_config_CNN_MNIST,
    'beta_clipping_config':beta_clipping_config_CNN_MNIST,
    'gamma_clipping_percentage_config':gamma_clipping_percentage_config_CNN_MNIST,
    'gamma_clipping_config':gamma_clipping_config_CNN_MNIST,
    'beta_gamma_config':beta_gamma_config_CNN_MNIST,
}
Config_model_list.append(config_CNN_MNIST)

config_MLP_MNIST={
    'name':'MNIST_model_MLP',
    'beta_config':beta_config_MLP_MNIST,
    'gamma_config':gamma_config_MLP_MNIST,
    'clipping_percentage_config':clipping_percentage_config_MLP_MNIST,
    'clipping_config':clipping_config_MLP_MNIST,
    
    'beta_clipping_percentage_config':beta_clipping_percentage_config_MLP_MNIST,
    'beta_clipping_config':beta_clipping_config_MLP_MNIST,
    'gamma_clipping_percentage_config':gamma_clipping_percentage_config_MLP_MNIST,
    'gamma_clipping_config':gamma_clipping_config_MLP_MNIST,
    'beta_gamma_config':beta_gamma_config_MLP_MNIST,
}
Config_model_list.append(config_MLP_MNIST)