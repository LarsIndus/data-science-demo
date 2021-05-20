import pandas as pd
import numpy as np

# Plotting
import seaborn as sns
from matplotlib import pyplot as plt

# Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Model selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve


# 1. Evaluate model on training set -------------------------------------------
def evaluate_on_training_set(model, X_train: pd.DataFrame,
                             y_train: pd.Series, cv: int = 5) -> None:
    
    """
    Fits a model on training data, evaluates performance on the training data
    and estimates how well it generalizes by performing cross-validation.
    
    model   -- sklearn regression model
    X_train -- pd.DataFrame; features of training data
    y_train -- pd.Series; target values of training data
    cv      -- int; number of folds for cross-valdidation
    """
    
    # Fit model and make predictons
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    
    # If predictions are smaller than 0, replace by 0
    train_predictions = np.where(train_predictions < 0, 0, train_predictions)
    
    # Calculate and print metrics
    MAE = mean_absolute_error(y_train, train_predictions)
    RMSE = mean_squared_error(y_train, train_predictions, squared = False)
    
    print(f"MAE (training): {MAE:.2f}")
    print(f"RMSE (training): {RMSE:.2f}")
    
    # Print validation error if CV has been done
    if cv is not None:
        # Scoring method is MAE
        cv_score = cross_val_score(model, X_train, y_train, cv = cv,
                                   scoring = "neg_mean_absolute_error")
        print(f"MAE cross-validated ({d} folds): {-np.mean(cv_score):.2f}")


# 2. Learning curves ----------------------------------------------------------

def learning_curves(model, X_train: pd.DataFrame, y_train: pd.Series,
                    cv: int = 5, training_splits: int = 5) -> None:
    
    """
    Plot learning curves to estimate how a model's learning performance
    changes with increasing number of training samples.
    
    model           -- sklearn regression model
    X_train         -- pd.DataFrame; features of training data
    y_train         -- pd.Series; target values of training data
    cv              -- int; number of folds for cross-valdidation
    training_splits -- int; number of training set sizes
    """
    
    # Evenly split array from 1 to maximum available training samples;
    # number of points equals training_splits.
    # As we are using cross validation,
    # we can only use (cv - 1) / cv of the training set.
    train_sizes = np.linspace(1, X_train.shape[0] * (cv - 1) / cv,
                              num = training_splits, dtype = "int")
    
    # Extract the learning scores
    train_sizes, train_scores, validation_scores = learning_curve(
        model, X_train, y_train,
        train_sizes = train_sizes,
        cv = cv,
        scoring = "neg_mean_absolute_error",
        shuffle = True
    )
    
    # Prepare a dataframe for plotting
    plot_df = pd.DataFrame({
        "Train Size" : train_sizes,
        "Training Error" : -train_scores.mean(axis = 1),
        "Validation Error" : -validation_scores.mean(axis = 1)
    })
    
    # Create the plot
    g = sns.lineplot(x = "Train Size", y = "value", hue = "variable",
                     data = pd.melt(plot_df, ["Train Size"]))
    g.set(
        title = "Learning Curves for " + model.__class__.__name__,
        xlabel = "Training Size", ylabel = "MAE"
    )
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles = handles[0 : ], labels = labels[0 : ])


# 3. Evaluate model on test set -----------------------------------------------

def evaluate_on_test_set(model, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series) -> None:
    
    """
    Fits a model on training data, evaluates performance on the test data
    and plots predicted against real values.
    
    model   -- sklearn regression model
    X_train -- pd.DataFrame; features of training data
    y_train -- pd.Series; target values of training data
    X_test  -- pd.DataFrame; features of test data
    y_test  -- pd.Series; target values of test data
    """
    
    # Fit model and make predictons
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
        
    # If predictions are smaller than 0, replace by 0
    y_pred = np.where(y_pred < 0, 0, y_pred)

    # Calculate and print metrics
    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = mean_squared_error(y_test, y_pred, squared = False)
    
    print(f"MAE: {MAE:.2f}")
    print(f"RMSE: {RMSE:.2f}")
        
    # Plot real against predicted values
    plt.figure(figsize = (7, 7))

    a = plt.axes(aspect = "equal")
    axis_max = max(max(y_test), max(y_pred))
    lims = [-0.05 * axis_max, 1.05 * axis_max]
    _ = plt.plot(lims, lims, color = "red")

    g = sns.scatterplot(x = y_test, y = y_pred)
    g.set(
        title = "Real vs. Predicted Values",
        xlabel = "Real Values", ylabel = "Predicted Values",
        xlim = lims,
        ylim = lims
    )