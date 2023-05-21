import pandas as pd
import numpy as np


class CustomPreprocessor:

    def __init__(self, df, transfer, ft='fit'):
        self.df = df
        print('*************', self.df.isnull().any(axis=1).sum(), 'variables have NaNs')
        self.proc_df_p1 = None
        self.variables_removed_in_correlation_test_p1 = transfer['var_corr']
        self.float_cols_initial_p2 = transfer['flo_init']
        self.int_cols_initial_p2 = transfer['int_init']
        self.floats_from_disc_p3 = transfer['flo_disc']
        self.bool_from_disc_p3 = transfer['bin_disc']
        self.disc_from_disc_p3 = transfer['dis_disc']
        self.dummy_threshold_p4 = 4
        self.dummy_disc_df_p4 = pd.DataFrame()
        self.fourcut_p4 = transfer['fourcut']
        self.twocut_p4 = transfer['twocut']
        self.bool_df_p5 = pd.DataFrame()
        self.binary_cols_to_be_removed_p5 = transfer['bin_rem']
        self.fit_or_transform = ft
        self.columns_with_one_value = transfer['val1_col']

    def run_fit(self):
        """
        Perform a series of cleaning steps on the input DataFrame and return a preprocessed DataFrame.

        Cleaning steps:
        1. Remove rows with missing values and correlated columns.
        2. Split the remaining columns into floating-point and integer columns, and save them globally to `self`.
        3. Split the columns into boolean, discrete, and floating-point based on likelihood and unique values.
        4. Convert the discrete columns to dummies (binary) using two different functions.
        5. Remove columns with few `1` values from the boolean columns obtained in step 3.
        6. Collect the floating-point columns obtained in steps 2 and 3.
        7. Create a DataFrame of the floating-point columns.
        8. Create a boolean DataFrame from the binary columns obtained in step 4 and the boolean columns obtained in step 5.
        9. Concatenate all the floating-point and boolean columns.

        Returns:
            A preprocessed DataFrame.
        """

        # Perform the first cleaning step: remove rows with missing values and correlated columns
        self.p1_pre_clean()

        # Perform the second cleaning step: split the remaining columns into floating-point and integer columns,
        # and save them globally to `self`.
        self.p2_sep_disc_float_initial()

        # Perform the third cleaning step: split the columns into boolean, discrete, and floating-point based on
        # likelihood and unique values.
        self.p3_sep_types_likely()

        # Perform the fourth cleaning step: convert the discrete columns to dummies (binary) using two different functions.
        self.p4_disc_converter()

        # Perform the rest of the cleaning steps.
        preprocessed_df = self.p5_finalize()

        # Check that the preprocessed DataFrame has non-zero dimensions
        assert (0 in preprocessed_df.shape) is False, 'At the end of preprocessing one dimension of X is zero!'

        # Return the preprocessed DataFrame
        return preprocessed_df

    def run_transform(self):

        # Print information about the initial column types and other metadata
        print('======\n',
              '\nfloatinit', self.float_cols_initial_p2,
              '\nintinit', self.int_cols_initial_p2,
              '\nFLOATDISC', self.floats_from_disc_p3,
              '\nboolDISC', self.bool_from_disc_p3,
              '\ndiscdisc', self.disc_from_disc_p3,
              '\novaval', self.columns_with_one_value)
        print('=====\n')

        # Numerize the DataFrame
        self.df2 = self.numerize(self.df)

        # Remove correlated variables and fill missing values with median
        self.proc_df_p1 = self.df2.drop(self.variables_removed_in_correlation_test_p1, axis=1)
        self.proc_df_p1.fillna(self.proc_df_p1.median(), inplace=True)

        # Convert discrete variables
        self.p4_disc_converter()

        # Finalize the preprocessing
        preprocessed_df = self.p5_finalize()

        # Check that the preprocessed DataFrame has non-zero dimensions
        assert (0 in preprocessed_df.shape) is False, 'At the end of preprocessing one dimension of X is zero!'

        # Return the preprocessed DataFrame
        return preprocessed_df

    """
    functions below
    """
    def kick_corr(self, in_df: pd.DataFrame, corr_th: float = 0.85) -> pd.DataFrame:
        """
        Removes correlated features above a given threshold.

        :param in_df: The input DataFrame of features.
        :param corr_th: The correlation threshold to use. Default is 0.95.
        :return: The DataFrame without correlated features.
        """
        # Compute the correlation matrix using Spearman's rank correlation coefficient.
        corr_matrix = in_df.corr('spearman').abs()

        # Create a mask to only include the upper triangle of the correlation matrix.
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find columns with correlations above the threshold.
        to_drop = [column for column in upper.columns if any(upper[column] > corr_th)]

        # Drop the correlated columns from the DataFrame.
        no_corr_df = in_df.drop(to_drop, axis=1)

        return no_corr_df

    def try_num(self, vec):
        """
        Try to convert a vector to a numeric type. If successful, return the converted vector,
        else return the original vector.

        :param vec: Pandas Series to convert
        :return: Converted numeric series or original series
        """
        try:
            get_f = pd.to_numeric(vec, downcast='integer', errors='ignore')
            return get_f
        except:
            return vec

    def numerize(self, x_in):
        """
        Converts non-numeric columns of a DataFrame into numeric ones.

        :param x_in: input DataFrame with non-numeric and numeric columns
        :type x_in: pandas.DataFrame

        :return: a DataFrame with all columns converted to numeric, where possible
        :rtype: pandas.DataFrame
        """
        # TODO could thrwo an error here train vs test set
        # Convert all columns to numeric using try_num function
        x_in = x_in.apply(self.try_num)

        # Select columns that are already numeric
        x_num = x_in.select_dtypes(include=[np.number])
        print(x_num.shape, 'is numeric from', x_in.shape)

        # Define a function to evaluate each non-numeric cell to convert it to numeric
        def eval_each(x):
            """
            Helper function that evaluates each non-numeric cell and converts it to numeric.

            :param x: input cell
            :type x: object

            :return: converted numeric cell or np.nan if it couldn't be converted
            :rtype: float or np.nan
            """
            try:
                return float(x)
            except:
                try:
                    return float(eval(x))
                except:
                    return np.nan

        # Select non-numeric columns
        x_non_num = x_in.select_dtypes(include='object')
        print(x_non_num.shape, 'was not numeric')

        # Apply eval_each function to each cell of the non-numeric columns
        x_non_num = x_non_num.applymap(eval_each)
        print(x_non_num.select_dtypes(include=[np.number]).shape,
              'is numeric now again')

        # Concatenate numeric and converted non-numeric columns
        x_out = pd.concat([x_non_num.select_dtypes(include=[np.number]),
                           x_num], axis=1)

        return x_out

    def p1_pre_clean(self):
        """
        Clean the input data by converting bad values to NaN and dropping columns with NaNs.
        Then remove any correlated features using the "kick_corr" function.

        :return: None
        """

        # Convert bad values to NaNs and convert non-numeric columns to numeric if possible
        self.df = self.numerize(self.df)

        # Check if any dimension of the input dataframe is zero
        assert (0 in self.df.shape) is False, 'One dimension of the input dataframe is zero!'

        # TODO fillna
        # Replace NaNs with the median value of each column
        thresh_for_nons = int(len(self.df)*.8)
        self.df.dropna(thresh = thresh_for_nons, axis=1, inplace=True)
        df_rem_nans = self.df.fillna(self.df.median())

        # Remove any correlated features using the "kick_corr" function
        self.proc_df_p1 = self.kick_corr(df_rem_nans)

        # Record the variables that were removed in the correlation test
        self.variables_removed_in_correlation_test_p1 = \
            list(set(self.proc_df_p1.columns).symmetric_difference(set(df_rem_nans.columns)))

    def p2_sep_disc_float_initial(self):
        """
        Tries to convert data to integer. Separates floats from ints.

        :return: A list of floats and a list of integers.
        """

        # Convert columns to integers if possible.
        self.proc_df_p1 = self.proc_df_p1.apply(lambda x: pd.to_numeric(x, downcast='integer'))

        #TODO what if a few floats
        #TODO object
        # Get the names of float columns and integer columns.
        self.float_cols_initial_p2 = self.proc_df_p1.select_dtypes(include='float').columns.tolist()
        self.int_cols_initial_p2 = self.proc_df_p1.select_dtypes(include='integer').columns.tolist()

    @staticmethod
    def get_bool_df(in_df):
        """
        Returns a list of columns in the input DataFrame that contain boolean values.

        :param in_df: input DataFrame
        :return: tuple of two lists - the first containing the boolean columns and the
        second containing the non-boolean columns
        """
        # Create a list of columns where all non-NaN values are either 0 or 1
        bool_cols = [col for col in in_df.columns if in_df[col].dropna().value_counts().index.isin([0, 1]).all()]

        # Create a list of columns that are not boolean
        non_binary = list(set(in_df.columns).symmetric_difference(set(bool_cols)))

        return bool_cols, non_binary

    def p3_sep_types_likely(self, thresh_categorical=.05):
        """
        Separates columns on variable types based on likelihood of belonging to a certain type.
        It checks whether there are more unique values than given by threshold, here 5%.

        :param thresh_categorical: (float) Threshold (in decimal) for percentage of uniques in column (default: 0.05)
        :return: (tuple) Three lists - boolean columns, discrete columns, and float columns.
        """

        # create dictionary to store whether columns are likely to be categorical
        dict_likely_categorical = {}

        # get the integer columns from proc_df_p1
        current_df = self.proc_df_p1[self.int_cols_initial_p2]

        # check if there are any null values in current_df
        assert (current_df.isnull().sum().any() == False), 'in p3 Nans'

        # check each integer column and mark it as likely categorical or not based on the threshold
        for var in current_df.columns:
            dict_likely_categorical[var] = str(
                1. * current_df[var].nunique() / current_df[var].count() < thresh_categorical)
            # TODO instead of theshold maybe just some integer
            # TODO maybe predefine cat variables

        # create a dataframe with column names and whether they're likely to be categorical or not
        likely_categorical_df = pd.DataFrame.from_dict(dict_likely_categorical, orient='index', columns=['tf'])

        # separate columns into those that are likely to be categorical and those that are likely not to be
        self.floats_from_disc_p3 = likely_categorical_df[likely_categorical_df.tf == 'False'].index.tolist()
        likely_cat_from_disc = likely_categorical_df[likely_categorical_df.tf == 'True'].index.tolist()

        # separate boolean and discrete columns from the likely categorical columns
        self.bool_from_disc_p3, self.disc_from_disc_p3 = self.get_bool_df(current_df[likely_cat_from_disc])

    def cut_2_dummies(self, in_df, var) -> pd.DataFrame:
        """
        Create dummy variables for a variable with less than 3 unique values using binary cut.
        :param in_df: DataFrame containing the variable
        :param var: variable to be dummified
        :return: DataFrame with dummy columns
        """
        # If we're in "fit" mode, create the binary cut and store the bins.
        if self.fit_or_transform == 'fit':
            to_dummy, bins2 = pd.cut(in_df[var],
                                     bins=2, labels=['b0', 'b1'],
                                     right=False, retbins=True)
            self.twocut_p4.update({var: bins2})
        # If we're in "transform" mode, use the previously-stored bins to create the binary cut.
        else:
            to_dummy = pd.cut(in_df[var],
                              bins=self.twocut_p4[var],
                              labels=['b0', 'b1'], right=False)
        # Create dummy variables using pd.get_dummies().
        return pd.get_dummies(to_dummy, prefix=var)

    def cut_4_dummies(self, in_df, var) -> pd.DataFrame:
        """
        Cuts a continuous variable into 4 bins and returns 4 dummy columns.
        :param in_df: The input DataFrame with the continuous variable to be cut and dummified.
        :param var: The name of the continuous variable to be cut and dummified.
        :return: A DataFrame with 4 dummy columns.
        """

        # Define the labels for the 4 dummy columns.
        c1_c4 = ['c1', 'c2', 'c3', 'c4']

        # If the function is in 'fit' mode, cut the continuous variable into 4 bins and return the labels of the bins.
        if self.fit_or_transform == 'fit':
            to_dummy, bins = pd.cut(in_df[var], bins=4, labels=c1_c4, right=False, retbins=True)
            self.fourcut_p4.update({var: bins})
        # If the function is in 'transform' mode, cut the continuous variable into 4 bins using the labels of the bins from the 'fit' mode.
        else:
            to_dummy = pd.cut(in_df[var], bins=self.fourcut_p4[var], labels=c1_c4, right=False)

        # Return the dummy columns with the corresponding labels.
        return pd.get_dummies(to_dummy, prefix=var)

    def p4_disc_converter(self):
        """
        Converts given columns in a DataFrame to dummy variables using either the "cut_4_dummies" or "cut_2_dummies"
        function based on the number of unique values in the columns. The resulting dummy columns are concatenated
        into a single DataFrame.

        :return: A DataFrame containing the dummy columns for the discrete variables.
        """
        # TODO should be configurable
        # Iterate over each discrete column
        for col in self.disc_from_disc_p3:
            if self.fit_or_transform=="fit":
                # Check the number of unique values in the column
                if self.proc_df_p1[col].nunique() > self.dummy_threshold_p4:
                    # Cut the variable into four bins and create four dummy columns using the "cut_4_dummies" function
                    temp_df_p4 = self.cut_4_dummies(in_df=self.proc_df_p1, var=col)
                else:
                    # Cut the variable into two bins and create two dummy columns using the "cut_2_dummies" function
                    temp_df_p4 = self.cut_2_dummies(in_df=self.proc_df_p1, var=col)
            else:
                if col in self.twocut_p4:
                    temp_df_p4 = self.cut_2_dummies(in_df=self.proc_df_p1, var=col)
                else:
                    temp_df_p4 = self.cut_4_dummies(in_df=self.proc_df_p1, var=col)

            # Concatenate the dummy columns generated for the current variable to the dummy_disc_df_p4 DataFrame
            self.dummy_disc_df_p4 = pd.concat([self.dummy_disc_df_p4, temp_df_p4], axis=1)

        # Check that all the values in the dummy columns are either 0 or 1
        assert self.dummy_disc_df_p4.apply(lambda x: x.value_counts()).index.isin([0, 1]).all() == True

    def p5_finalize(self) -> pd.DataFrame:
        """
        Finalizes the preprocessing pipeline and returns the processed DataFrame.
        Drops columns with all NaN values and columns with only one unique value.
        :return: preprocessed DataFrame
        """

        # Concatenate float and boolean DataFrames
        processed_df = pd.concat([
            self.proc_df_p1[self.float_cols_initial_p2 + self.floats_from_disc_p3],
            self.bool_df_p5
        ], axis=1)

        # Drop columns with all NaN values
        print('=====nan columns', processed_df.isnull().sum().sort_values())
        processed_df.dropna(how='all', axis=1, inplace=True)

        # Drop columns with only one unique value
        self.columns_with_one_value = processed_df.loc[:, processed_df.apply(pd.Series.nunique) == 1].columns
        processed_df.drop(self.columns_with_one_value, axis=1, inplace=True)

        # Check if there are still any NaN values
        assert (processed_df.isnull().sum().any() == False), 'Preprocessed DataFrame still has NaNs'

        return processed_df

