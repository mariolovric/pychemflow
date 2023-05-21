from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from core.preprocessor import CustomPreprocessor
from sklearn import set_config
set_config(transform_output = "pandas")

class ProcTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.variables_removed_in_correlation_test_p1 = []
        self.float_cols_initial_p2 = []
        self.int_cols_initial_p2 = []
        self.floats_from_disc_p3 = []
        self.bool_from_disc_p3 = []
        self.disc_from_disc_p3 = []
        self.fourcut_p4 = []
        self.twocut_p4 = []
        self.binary_cols_to_be_removed_p5 = []
        self.fit_or_transform = None
        self.columns_with_one_value = []
        self.transfer_dict = {}

    def fit(self, X, y=None):
        """
        Fits the CustomModel to the input data.

        :param X : array-like or sparse matrix of shape (n_samples, n_features) - The input data to fit the model to.
        :param y : array-like of shape (n_samples,), default=None - The target variable to fit the model to. If None, unsupervised learning is performed.
        :return self : object - Returns the instance itself.
        """
        # Step 1: Preprocess the input data
        # Instantiate a CustomPreprocessor object with the input data and a dictionary of transfer parameters
        p = CustomPreprocessor(X,
                               transfer={'var_corr': [], 'flo_init': [], 'int_init': [],
                                         'flo_disc': [], 'bin_disc': [], 'dis_disc': [],
                                         'fourcut': {}, 'twocut': {}, 'bin_rem': [], 'val1_col': []},
                               ft='fit')
        # Run the fit method of the CustomPreprocessor object to preprocess the input data
        Xa = p.run_fit()

        # Step 2: Store the results of the preprocessing in a transfer dictionary
        self.transfer_dict = {'var_corr': p.variables_removed_in_correlation_test_p1,
                              # Variables removed in correlation test (Part 1)
                              'flo_init': p.float_cols_initial_p2,  # Initial float columns (Part 2)
                              'int_init': p.int_cols_initial_p2,  # Initial integer columns (Part 2)
                              'flo_disc': p.floats_from_disc_p3,  # Floats derived from discrete columns (Part 3)
                              'bin_disc': p.bool_from_disc_p3,
                              # Binary variables derived from discrete columns (Part 3)
                              'dis_disc': p.disc_from_disc_p3,
                              # Discrete variables derived from discrete columns (Part 3)
                              'fourcut': p.fourcut_p4,  # Columns selected for 4-cut discretization (Part 4)
                              'twocut': p.twocut_p4,  # Columns selected for 2-cut discretization (Part 4)
                              'bin_rem': p.binary_cols_to_be_removed_p5,  # Binary columns to be removed (Part 5)
                              'val1_col': p.columns_with_one_value}  # Columns with only one unique value

        # Step 3: Return the instance of the class
        return self

    def transform(self, X, y=None):
        """
        :param X : array-like or sparse matrix of shape (n_samples, n_features) - The input data to fit the model to.
        :param y : array-like of shape (n_samples,), default=None - The target variable to fit the model to. If None, unsupervised learning is performed.
        :return Xb : array-like or sparse matrix of shape (n_samples, n_features) - The transformed input data.
        """
        # Step 1: Preprocess the new input data using the stored transfer dictionary
        # Instantiate a CustomPreprocessor object with the new input data and the stored transfer dictionary
        p = CustomPreprocessor(X, transfer=self.transfer_dict, ft='transform')
        # Run the transform method of the CustomPreprocessor object to preprocess the new input data
        Xb = p.run_transform()

        # Step 2: Return the transformed input data
        return Xb


def build_preprocess_pipeline():
    return Pipeline(
    steps=[
        ("use_custom_transformer", ProcTransformer()),
        ('scaler', StandardScaler()),
        ('var_threshold', VarianceThreshold(threshold=0.05))
    ]
)


preproc_pipe = build_preprocess_pipeline()


