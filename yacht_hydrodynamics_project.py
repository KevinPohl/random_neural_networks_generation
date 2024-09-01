# used versions for this assignment:
# python 3.11.9
# keras 2.15.0
# matplotlib 3.7.4
# numpy 1.25.2
# pandas 2.1.3
# scikit-learn 1.3.2
# seaborn 0.13.0
# tensorflow 2.15.0
# typing_extentions 4.12.2



from keras import Sequential
from keras.layers import Dense, InputLayer
from keras.metrics import MeanSquaredError
from keras.losses import MeanSquaredError
from keras.optimizers import SGD # , Adadelta, AdamW, Adam
from matplotlib.pyplot import show, subplots
from numpy import array, ceil, floor, sqrt
from numpy.random import seed as np_seed
from os import environ
from pandas import DataFrame, read_csv
from random import choice, randint, seed as rd_seed, getstate as rd_getstate, setstate as rd_setstate
from sklearn.preprocessing import RobustScaler # , StandardScaler
from seaborn import pairplot
import tensorflow as tf
from typing import Optional

def set_seeds(seed:int):
    """
    ### set_seeds function
    
    ---
    #### description
        Function to initialize seeds for all libraries which might have stochastic behavior
    
    ---
    #### input
        seed: int which is then used as seed
    
    ---
    #### returns
        None
    
    ---
    #### example
    ```python
    set_seeds(seed= seed)
    ```
    """
    environ['PYTHONHASHSEED'] = str(seed)
    rd_seed(seed)
    tf.random.set_seed(seed)
    np_seed(seed)

def set_global_determinism(seed:int):
    """
    ### set_global_determinism function
    
    ---
    #### description
        sets seeds via set_seeds and activates Tensorflow deterministic behavior
    
    ---
    #### input
        seed: int which is then used as seed
    
    ---
    #### returns
        None
    
    ---
    #### example
    ```python
    set_global_determinism(seed= seed)
    ```
    """
    set_seeds(seed=seed)

    environ['TF_DETERMINISTIC_OPS'] = '1'
    environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # one environment change to turn off rounding errors in tensorflow
    environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)
    pass

def print_nn_collection(collection):
    """
    ### print_nn_collection function
    
    ---
    #### description
        plots an arbitrary number of neural network returns into self organized subplots, ordered in z-logic
    
    ---
    #### input
        collection: a list of dicts with this minimal fields: "history", "setup", "optimizer", "loss", "alpha", "epochs", "batch_size", "seed"
    
    ---
    #### returns
        None
    """
    if not isinstance(collection, list):
        collection = [collection]
    cols, rows = dim_subplots(len(collection))
    fig, ax = subplots(nrows= rows, ncols= cols, sharex= True, squeeze= True)

    if len(fig.axes) == 1:
        ax = array(ax, ndmin= 2)
    elif len(ax.shape) == 1:
        ax = array(ax, ndmin= 2)

    fig.set_size_inches(16, 8)
    title = f'different neural nets in comparison\n'
    title += f'epochs: {collection[0]["epochs"]}, '
    title += f'alpha: {collection[0]["alpha"]}, '
    title += f'batch size: {collection[0]["batch_size"]}, '
    title += f'seed: {collection[0]["seed"]}'

    fig.suptitle(title)

    for idx, model in enumerate(collection):
        axis = ax[axwalker_z(idx, cols)]
        axis.plot(model["history"].history["loss"], label="train")
        axis.plot(model["history"].history["val_loss"], label="test")
        axis.set_title(f'{model["setup"]}')
        axis.set_ylabel("loss")
        axis.set_xlabel("epoch")
        axis.set_ybound(lower= -0.01)
        if idx == 0:
            fig.legend()
        pass
    show()
    pass

def _parse_hidden_setup(model, hidden_setup):
    """
    ### _parse_hidden_setup function
    
    ---
    #### description
        adds layers parsed from list of tuples to a neural network sequential model
    
    ---
    #### input
        model: the model to add the layer to
        hidden_setup: the hidden layer setup to be parsed and added to model
    
    ---
    #### returns
        None
    
    ---
    #### example
    ```python
    >>> _parse_hidden_setup(model1, [(7, None), (12, 'relu')])
    >>> # adds layers to model1
    >>> model1.add(Dense(units= 7))
    >>> model1.add(Dense(units= 12, activation= 'relu'))
    ```
    """
    for layer in hidden_setup:
        if isinstance(layer, tuple):
            if len(layer) == 2:
                model.add(Dense(units= layer[0], activation= layer[1]))
            else:
                raise Exception("unknown hidden setup format")
        else:
            model.add(Dense(units=layer))
    pass

def configure_nn(
    X,
    y,
    hidden_setup,
    alpha,
    seed:Optional[int],
    epochs:int= 1,
    batch_size:int = 1,
    optimizer = "SGD",
    loss = MeanSquaredError(name="MSE"),
    validation_split:float = 0.2
    ):
    """
    ### configure_nn function
    
    ---
    #### description
        compiles and fits a neural network model
    
    ---
    #### input
        X: DataFrame of feature data, scaled beforehand if necessary
        y: DataFrame or Series of label data
        hidden_setup: the hidden layer setup to be parsed and added to model
        alpha: float of alpha, or learning rate
        seed: int which is then used as seed
        epochs: int as count of training reruns 
        batch_size: int as batch size
        optimizer: string or object of tensorflow optimizer
        loss: string or object of tensorflow loss
        validation_split: float of percentage of validation to training data split
    
    ---
    #### returns
        a dict with this minimal fields: "history", "setup", "optimizer", "loss", "alpha", "epochs", "batch_size", "seed" representing the results and attributes of a neural network
    
    ---
    #### example
    ```python
    >>> model = configure_nn(X, y, [(12, 'relu'), (1, None)], 0.003, 1337)
    >>> print(model)
    {'history': <keras.src.callbacks.History object at 0x00000256219FFBD0>, 'setup': "[(12, 'relu'), (1, None)]", 'optimizer': 'SGD', 'loss': 'MSE', 'alpha': 0.003, 'epochs': 1, 'batch_size': 1, 'seed': 1337}
    >>> print(model["setup"])
    [(12, 'relu'), (1, None)]
    ```
    """
    model = Sequential()
    # set input layer first
    model.add(InputLayer(input_shape= X.shape[1]))

    # parse hidden and output layer
    _parse_hidden_setup(model, hidden_setup)

    # set learning rate if standard loss is used
    if optimizer == "SGD":
        optimizer = SGD(learning_rate= alpha)
    
    # compile model
    model.compile(optimizer= optimizer, loss= loss)

    # reapply seeds
    set_seeds(seed)

    # save model fit results into a dictionary
    history = model.fit(X.values, y.values, batch_size= batch_size,
                        epochs= epochs, verbose= 0,
                        validation_split= validation_split)
    result = {"history": history, "setup": str(hidden_setup), "optimizer":
              model.optimizer.name, "loss": model.loss.name, "alpha": alpha,
              "epochs": epochs, "batch_size": batch_size, "seed": seed}
    return result
    
def bulk_fit(X, y, node_setups, learning_rate, epochs, batch_size):
    """
    ### bulk_fit function
    
    ---
    #### description
        fits and collects an arbitrary number of neural network models into a collection
    
    ---
    #### input
        X: DataFrame of feature data, scaled beforehand if necessary
        y: DataFrame or Series of label data
        node_setups: list of lists of tuples, representing a list of setups of neural network layers
        learning_rate: float of alpha, or learning rate
        epochs: int as count of training reruns 
        batch_size: int as batch size
    
    ---
    #### returns
        a collection of fitted neural network models 
    
    ---
    #### example
    ```python
    >>> collection = ([(4, "relu"), 1], [(4, "sigmoid"), 1])
    >>> models = bulk_fit(X, y, collection, 0.003, 1000, 128)
    >>> for idx, model in enumerate(models):
    >>>     print(f'[{idx}]: {model["setup"]}')
    [0]: [(4, 'relu'), 1]
    [1]: [(4, 'sigmoid'), 1]
    ```
    """
    models = list()

    for setup in node_setups:
        models.append(
            configure_nn(X, y, setup, learning_rate, seed, epochs,batch_size)
            )

    return models

def dim_subplots(n_plots:int= 1):
    """
    ### dim_subplots function
    
    ---
    #### description
        calculates the rows and columns needed to accomodate n_plots
    
    ---
    #### input
        n_plots: count of subplots to be plotted
    
    ---
    #### returns
        a tuple of a number of columns and rows needed to accomodate all subplots on a plot
    
    ---
    #### example
    ```python
    >>> print(dim_subplots(n_plots= 17))
    (5, 4)
    ```
    """
    if n_plots <= 0:
        return (0, 0)
    sqrt_len = sqrt(n_plots)
    columns = (sqrt_len != floor(sqrt_len)) + floor(sqrt_len)
    rows = ceil(n_plots / columns)
    return (int(columns), int(rows))

def axwalker_z(index:int, cols:int):
    """
    ### axwalker_z function
    
    ---
    #### description
        walks subplots ordered in columns and rows, both with indexing starting at 0, and in z-logic order, given the number of columns and the index
    
    ---
    #### input
        index: int of subplot to be addressed, starting at 0
        cols: number of columns in plot, starting at 0
    
    ---
    #### returns
        a tuple of the row and column as location of the subplot in the plot
    
    ---
    #### example
    ```python
    >>> print(axwalker_z(index= 17, cols= 5))
    (3, 2)
    ```
    """
    col = int((index % cols))
    row = int(floor(index / cols))
    return (row, col)

def rnd_setups(n_setups:int= 1,
               min_nodes:int= 1,
               max_nodes:int= 100,
               min_layers: int = 1,
               max_layers:int= 5,
               fixed_output:Optional[tuple]=None,
               activations:list = ["relu", "sigmoid", "tanh", None],
               ignore_seed:bool=False):
    """
    ### rnd_setups function
    
    ---
    #### description
        Creates a count of n_setups of random neural network setups in depth and length, with random activation functions and an optional fixed output layer. For this, a set seed can optionally be ignored. Input layer is ignored, as it is set up elsewhere based on calculated feature count.
        
    ---
    #### input
        n_setups: int as number of randomly generated neural network setups
        min_nodes: int as minimum assigned nodes to each layer
        max_nodes: int as maximum assigned nodes to each layer
        min_layers: int as minimum layers in neural network, excluding input layer but including output layer even if fixed
        max_layers: int as maximum layers in neural network, excluding input layer but including output layer even if fixed
        fixed_output: fixed output layer in the form (5, 'relu') or (5, None)
        activations: a list of activation functions to randomly pick from
        ignore_seed: ignores the set seed for neural network creation, would otherwise construct the same neural nets if a seed is set
    
    ---
    #### returns
        * if ignore_seed is False, a list of lists of tuples, representing a list of setups of neural network layers
        * if ignore_seed is True, a tuple a list of lists of tuples and an int, representing a tuple of a list of setups of neural network layers and the temporarily created seed for neural network setup creation
    
    ---
    #### example
    ```python
    >>> seed = 1337
    >>> set_seeds(seed)
    >>> setups, temp_seed = rnd_setups(n_setups= 1, fixed_output= (1, None), ignore_seed= True)
    >>> print(f"setup: {setups}\\ntemporary seed: {temp_seed}")
    setup: [[(97, None), (33, None), (21, 'relu'), (1, None)]]
    temporary seed: 785096
    >>> setups = rnd_setups(n_setups= 1, fixed_output= (1, None), ignore_seed= False)
    >>> print(f"setup: {setups}\\nseed: {seed}")
    setup: [[(74, 'sigmoid'), (100, 'tanh'), (50, 'tanh'), (1, None)]]
    seed: 1337
    ```
    """
    # save and set seed to None to ignore
    if ignore_seed:
        ignored_seed = rd_getstate()
        rd_seed(None)
        seed = randint(1, 1e6)
        rd_seed(seed)
        pass

    if not isinstance(fixed_output, type(None)):
        max_layers -= 1

    setups = list()
    for _ in range(n_setups):
        layer = list()
        n_layers = randint(min_layers, max_layers)
        for _ in range(n_layers):
            n_nodes = randint(min_nodes, max_nodes)
            activation = choice(activations)
            layer.append((n_nodes, activation))
            pass
        if not isinstance(fixed_output, type(None)):
            if isinstance(fixed_output, tuple):
                layer.append(fixed_output)
            else:
                raise Exception("unknown fixed output, example is: (13, 'relu') or (4, None)")
        setups.append(layer)
        pass
    
    # reset seed after setup creation
    if ignore_seed:
        rd_setstate(ignored_seed)
        return setups, seed
    return setups

# ------------------------------------------------------------------------------

# data is from https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics
# Dataset Information
# Prediction of residuary resistance of sailing yachts at the initial design
# stage is of a great value for evaluating the ship's performance and for
# estimating the required propulsive power. Essential inputs include the basic
# hull dimensions and the boat velocity. 
# The Delft data set comprises 308 full-scale experiments, which were performed
# at the Delft Ship Hydromechanics Laboratory for that purpose. 
# These experiments include 22 different hull forms, derived from a parent form
# closely related to the 'Standfast 43' designed by Frans Maas.
# Additional Variable Information
# Variations concern hull geometry coefficients and the Froude number:

# 1. Longitudinal position of the center of buoyancy, adimensional.
# 2. Prismatic coefficient, adimensional.
# 3. Length-displacement ratio, adimensional.
# 4. Beam-draught ratio, adimensional.
# 5. Length-beam ratio, adimensional.
# 6. Froude number, adimensional.

# The measured variable is the residuary resistance per unit weight of
# displacement:
# 7. Residuary resistance per unit weight of displacement, adimensional.

# ------------------------------------------------------------------------------
# get data
df = read_csv("Project/yacht.csv", delimiter=" ", engine='python', quotechar="'")
# ------------------------------------------------------------------------------

# first data evaluation
print(df.shape)
print(df.sample(5))
print(df.describe(include="all"))
df.info()

skip_pairplot = True
if not skip_pairplot:
    pairplot(df, hue= "Residuary_Resist")
    show()

# define X and y
X = df.drop("Residuary_Resist", axis=1)
y = df["Residuary_Resist"]

# ------------------------------------------------------------------------------
# hyperparams
alpha = 0.003
epochs = 10000
batch_size = 128
seed = 1337

# prepare seeds and environment
set_global_determinism(seed)

scaling = True
if scaling:
    scaler = RobustScaler()
    X = DataFrame(scaler.fit_transform(X), columns= X.columns)

# generation of 36 neural nets
if False:
    temp_seed = seed
    set_seeds(308778)
    setups = rnd_setups(36, fixed_output= (1, None), ignore_seed=False)
    models = bulk_fit(
        X= X,
        y= y,
        node_setups= setups,
        learning_rate= 0.003,
        epochs= 1000,
        batch_size= 128)

    print(f"rnd_model_creation_seed: {temp_seed}")
    for i, model in enumerate(models):
        print(f'[{i}]: {model["setup"]}')

    print_nn_collection(models)
    set_seeds(temp_seed)

# generation of 25 neural nets
if False:
    temp_seed = seed
    set_seeds(469267)
    setups = rnd_setups(25, fixed_output= (1, None), ignore_seed=False)
    models = bulk_fit(
        X= X,
        y= y,
        node_setups= setups,
        learning_rate= 0.003,
        epochs= 10000,
        batch_size= 128)

    print(f"rnd_model_creation_seed: {temp_seed}")
    for i, model in enumerate(models):
        print(f'[{i}]: {model["setup"]}')

    print_nn_collection(models)
    set_seeds(temp_seed)


# a closer look on a collection of well performing neural networks
if False:
    manual_setups = [
        [(69, 'sigmoid'), (24, None), (28, 'sigmoid'), (1, None)],
        [(97, 'sigmoid'), (10, 'tanh'), (1, None)],
        [(96, 'sigmoid'), (89, 'sigmoid'), (1, None)],
        [(15, 'sigmoid'), (1, None)],
        [(79, 'sigmoid'), (86, 'relu'), (38, 'sigmoid'), (50, 'sigmoid'), (1, None)],
        [(84, 'sigmoid'), (25, None), (54, 'relu'), (1, None)]
        ]

    for man_alpha in [3e-4, 3e-3, 3e-2, 3e-1]:
        models = bulk_fit(X, y, manual_setups, man_alpha, 10000, 128)
        print(f"alpha: {man_alpha}")
        for i, model in enumerate(models):
            print(f'[{i}]: {model["setup"]}')
        print_nn_collection(models)


# picked best two and closer range around last best alpha of 3e-4
if False:
    manual_setups = [
        [(96, 'sigmoid'), (89, 'sigmoid'), (1, None)],
        [(15, 'sigmoid'), (1, None)]
        ]

    for man_alpha in [9e-4, 8e-4, 7e-4, 6e-4]: # 5e-4, 4e-4, 3e-4, 2e-4, 1e-4
        models = bulk_fit(X, y, manual_setups, man_alpha, 10000, 128)
        print(f"alpha: {man_alpha}")
        for i, model in enumerate(models):
            print(f'[{i}]: {model["setup"]}')
        print_nn_collection(models)

# final neural network pick in great detail
if False:
    manual_setups = [
        [(15, 'sigmoid'), (1, None)]
        ]

    man_alpha = 5e-4

    models = bulk_fit(X, y, manual_setups, man_alpha, 50000, 128)
    print(f"alpha: {man_alpha}")
    for i, model in enumerate(models):
        print(f'[{i}]: {model["setup"]}')
    print_nn_collection(models)

# replot final neural network with 6 random seeds
if False:
    manual_setups = [
        [(15, 'sigmoid'), (1, None)]
        ]

    man_alpha = 5e-4

    # save random state from before
    temp_seed = rd_getstate()
    models = []

    # create 5 random seeds
    for i, seed in enumerate([randint(0, 1e6) for _ in range(6)]):

        # seed random random state
        set_seeds(seed)

        # status print: iteration, alpha, seed
        print(f"[{i}] - alpha: {man_alpha}, seed: {seed}")

        # compile and fit the neural network with random seed and fixed alpha, epochs, batch_size
        models.append(configure_nn(X, y, [(15, 'sigmoid'), 1], man_alpha, seed, 20000, batch_size))

    # print the results
    print_nn_collection(models)

    # reset random state to before
    rd_setstate(temp_seed)






pass # end of if __name__ == "__main__"
# ------------------------------------------------------------------------------



