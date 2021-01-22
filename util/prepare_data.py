import pandas as pd
from pathlib import Path

def prepare_data(raw_data: pd.DataFrame, categorical_cols: list[str] = None,
                 drop_cols: list[str] = None, bin_hours: bool = True,
                 bin_year_month: bool = True, drop_hour: bool = False,
                 drop_year_month: bool = False, rename: bool = True,
                 dummies: bool = True) -> pd.DataFrame:
    
    """
    Prepares the data (cleaning etc.)
    
        - convert certain columns to categorical
        - drop useless columns
        - (bin hours and possibly drop original column)
        - (bin years and months and possibly drop original columns)
        - rename some of the columns
        - create dummies for categorical variables
    
    raw_data          -- the data file
    categorical_cols  -- columns to be converted to categorical
    drop_cols         -- columns to be dropped
    bin_hours         -- should hours be binned?
    bin_year_month    -- should years and month be binned into one variable?
    drop_hour         -- drop original hours column (after binning)?
    drop_year_month   -- drop original year and month columns (after binning)?
    rename            -- rename columns?
    dummies           -- create dummies for categorical variables?
    """

    # work on a copy in order not to change the raw data
    data = raw_data.copy(deep = True)
    
    # drop columns
    data.drop(drop_cols, axis = 1, inplace = True)

    # Change some columns to category
    for var in categorical_cols:
        data[var] = data[var].astype("category")

    # Create hour bins?
    def _helper_bin_hours(row):
        if row["hr"] in [7, 8, 9, 17, 18, 19]:
            val = "high"
        if row["hr"] in [10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23]:
            val = "mid"
        if row["hr"] in [0, 1, 2, 3, 4, 5, 6]:
            val = "low"
        return val
    
    if bin_hours:
        data["hour_bin"] = data.apply(_helper_bin_hours, axis = 1)
        data["hour_bin"] = data["hour_bin"].astype("category")

    # Create quarter bins?
    def _helper_bin_quarters(row):
        if row["yr"] == 0 and row["mnth"] in [1, 2, 3]: val = "Q1_2011"
        if row["yr"] == 0 and row["mnth"] in [4, 5, 6]: val = "Q2_2011"
        if row["yr"] == 0 and row["mnth"] in [7, 8, 9]: val = "Q3_2011"
        if row["yr"] == 0 and row["mnth"] in [10, 11, 12]: val = "Q4_2011"
        if row["yr"] == 1 and row["mnth"] in [1, 2, 3]: val = "Q1_2012"
        if row["yr"] == 1 and row["mnth"] in [4, 5, 6]: val = "Q2_2012"
        if row["yr"] == 1 and row["mnth"] in [7, 8, 9]: val = "Q3_2012"
        if row["yr"] == 1 and row["mnth"] in [10, 11, 12]: val = "Q4_2012"
        return val
    
    if bin_year_month:
        data["quarter"] = data.apply(_helper_bin_quarters, axis = 1)
        data["quarter"] = data["quarter"].astype("category")

    # Do some renaming for better interpretability? (e.g. for plotting)
    if rename:    
        data["season"] = data["season"].cat.rename_categories({
            1 : "Spring", 2 : "Summer", 3 : "Fall", 4 : "Winter"})
        
        data["yr"] = data["yr"].cat.rename_categories({
            0 : "2011", 1 : "2012"})
        
        data["weekday"] = data["weekday"].cat.rename_categories({
            1 : "Monday", 2 : "Tuesday", 3 : "Wednesday", 4 : "Thursday",
            5 : "Friday", 6 : "Saturday", 0 : "Sunday"
        })
        data["weathersit"] = data["weathersit"].cat.rename_categories({
            1 : "clear", 2 : "misty", 3 : "light_rain", 4 : "heavy_rain"})

    if bin_hours and drop_hour:
            data.drop("hr", axis = 1, inplace = True)
    if bin_year_month and drop_year_month:
        data.drop(["yr", "mnth"], axis = 1, inplace = True)
    
    if dummies:
        data = pd.get_dummies(data, drop_first = True)
            
    return data

# Test the function (for debugging)
if __name__ == "__main__":
    from pathlib import Path

    file_path = Path(__file__).parent / '../data/hour.csv'
    raw_data = pd.read_csv(file_path)
    
    drop_cols = ["instant", "dteday", "atemp"]
    categorical_cols = ["season", "yr", "mnth", "hr", "holiday",
                        "workingday", "weekday", "weathersit"]
    
    data = prepare_data(
        raw_data = raw_data, categorical_cols = categorical_cols,
        drop_cols = drop_cols, bin_hours = True, bin_year_month = True,
        drop_hour = True, drop_year_month = True,
        rename = True, dummies = True)