# <em>PyChemFlow</em>
 An automated pre-processing pipeline in Python for reproducible machine learning on chemical data


## Installation of conda and the dependencies

conda create --name my_env python=3.10.9 <br>
conda activate my_env <br>
conda install scikit-learn=1.2.0 scipy=1.9.3 numpy=1.23.5 pandas=1.5.2 <br>

## Project structure

    .
    ├── data                 # Data for testing
    ├── saved_transformes    # The saved .joblib files for persistence
    ├── core                 # Source code
	├── test                 # Consists of a file showcasing usage
	├── LICENSE              # MIT license
    └── README.md            # Brief repo description and installation recommendation

## Run 
The file test/test.py is an example how to run the code.
	
	#load libraries
	import joblib
	import pandas as pd
	from core.transformer import preproc_pipe
	
	#load data
	df = pd.read_csv('<path>')

	#transform train set
	transformed_df = preproc_pipe.fit_transform(df)
	
	#save pipeline
	joblib.dump(preproc_pipe, '<path>')
	