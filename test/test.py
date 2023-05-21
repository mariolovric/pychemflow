# persist model
import joblib
import pandas as pd
from core.transformer import preproc_pipe

#load data
# load train set
df = pd.read_csv('../data/sol_features.csv', index_col=0)

#transform train set
# this is processed train set
transformed_df = preproc_pipe.fit_transform(df)
#dump pipeline
joblib.dump(preproc_pipe, '../saved_transformers/custom_tf.joblib')

#load data again for showcasing that it will undergo the same steps
# load test set
df1 = pd.read_csv('../data/sol_features.csv', index_col=0)

#load pipeline
transformer_loaded = joblib.load('../saved_transformers/custom_tf.joblib')
# this is processed test set
transformed_df2 = transformer_loaded.transform(df1)

print(type(transformed_df))
print(type(transformed_df2))
# showcasing that the transformer has succeeded by comparing the twice loaded and
# transformed file
print(transformed_df.equals(transformed_df2))
