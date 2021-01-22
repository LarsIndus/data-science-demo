from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer that does binning of variables
class BinningTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, bin_hours = True, bin_year_month = True):
        self.bin_hours = bin_hours
        self.bin_year_month = bin_year_month
        
    # Return self; nothing else to do here
    def fit(self, X, y = None):
        return self
    
    # Helper function to bin hours
    def _helper_bin_hours(self, row):
        if row["hr"] in [7, 8, 9, 17, 18, 19]:
            val = "high"
        if row["hr"] in [10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23]:
            val = "mid"
        if row["hr"] in [0, 1, 2, 3, 4, 5, 6]:
            val = "low"
        return val
    
    # Helper function to bin yr and month into quarters
    def _helper_bin_quarters(self, row):
        if row["yr"] == 0 and row["mnth"] in [1, 2, 3]: val = "Q1_2011"
        if row["yr"] == 0 and row["mnth"] in [4, 5, 6]: val = "Q2_2011"
        if row["yr"] == 0 and row["mnth"] in [7, 8, 9]: val = "Q3_2011"
        if row["yr"] == 0 and row["mnth"] in [10, 11, 12]: val = "Q4_2011"
        if row["yr"] == 1 and row["mnth"] in [1, 2, 3]: val = "Q1_2012"
        if row["yr"] == 1 and row["mnth"] in [4, 5, 6]: val = "Q2_2012"
        if row["yr"] == 1 and row["mnth"] in [7, 8, 9]: val = "Q3_2012"
        if row["yr"] == 1 and row["mnth"] in [10, 11, 12]: val = "Q4_2012"
        return val
    
    # Transformer method we wrote for this transformer
    def transform(self, X , y = None):
        
        X_trans = X.copy()
        
        # Depending on constructor argument creates new columns
        # using the helper functions
        if self.bin_hours:
            X_trans["hour_bin"] = X_trans.apply(
                self._helper_bin_hours, axis = 1)
            X_trans["hour_bin"] = X_trans["hour_bin"].astype("category")
            
        if self.bin_year_month:
            X_trans["quarter"] = X_trans.apply(
                self._helper_bin_quarters, axis = 1)
            X_trans["quarter"] = X_trans["quarter"].astype("category")
            
        return X_trans
    

# Custom transformer that drops columns not used anymore
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, drop_cols, verbose = False):
        self.drop_cols = drop_cols
        self.verbose = verbose
        
    # Return self; nothing else to do here
    def fit(self, X, y = None):
        return self

    # Transformer method we wrote for this transformer
    def transform(self, X , y = None):
        X_trans = X.copy()
        for col in self.drop_cols:
            if col in X_trans.columns:
                X_trans.drop(col, axis = 1, inplace = True)
            elif col not in X_trans.columns and self.verbose:
                print("DropTransformer: Could not drop '", col, "'; ",
                      "no such column. Ignore and continue ...", sep = '')
        return X_trans


# Custom transformer that changes column types
class DTypeTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, convert_to_category, verbose = False):
        self.convert_to_category = convert_to_category
        self.verbose = verbose
        
    # Return self; nothing else to do here
    def fit(self, X, y = None):
        return self
    
    # Transformer method we wrote for this transformer
    def transform(self, X , y = None):
        X_trans = X.copy()
        for col in self.convert_to_category:
            if col in X_trans.columns:
                X_trans[col] = X_trans[col].astype("category")
            elif col not in X_trans.columns and self.verbose:
                print("DTypeTransformer: Could not convert '", col, "'; ",
                      "no such column. Ignore and continue ...", sep = '')
        return X_trans


# Custom transformer that selects columns of a specified type
class TypeSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, include = None, exclude = None):
        self.include = include
        self.exclude = exclude
        
    # Return self; nothing else to do here
    def fit(self, X, y = None):
        return self
    
    # Transformer method we wrote for this transformer
    def transform(self, X , y = None):
        X_trans = X.copy()
        X_trans = X_trans.select_dtypes(
            include = self.include, exclude = self.exclude)
        return X_trans