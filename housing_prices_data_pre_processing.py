"""Missing values in numerical features will be imputed by replacing them with the median, as most ML algorithms don’t expect missing values. In categorical features, missing values will be replaced by the most frequent category.

The categorical feature will be one-hot encoded, as most ML algorithms only accept numerical inputs.

A few ratio features will be computed and added: bedrooms_ratio, rooms_per_house, and people_per_house. Hopefully these will better correlate with the median house value, and thereby help the ML models.

A few cluster similarity features will also be added. These will likely be more useful to the model than latitude and longitude.

Features with a long tail will be replaced by their logarithm, as most models prefer features with roughly uniform or Gaussian distributions.

All numerical features will be standardized, as most ML algorithms prefer when all features have roughly the same scale.
 """

# FEATURE ENGINEERING — Ratio function
# Creates a ratio between two numeric columns (e.g., bedrooms_ratio)
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]


# FEATURE ENGINEERING — Naming for ratio output column
# Ensures the new ratio feature has a proper name in the pipeline
def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # output column name

# FEATURE ENGINEERING + DATA CLEANING + FEATURE SCALING
# Builds a pipeline to:
# 1. Impute missing values with median (Data Cleaning)
# 2. Create a ratio feature (Feature Engineering)
# 3. Standardize to mean=0, std=1 (Feature Scaling)
def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())


# DATA CLEANING + FEATURE TRANSFORMATION + FEATURE SCALING
# Builds a pipeline to:
# 1. Impute missing values with median (Data Cleaning)
# 2. Apply log transform to skewed features (Feature Transformation)
# 3. Standardize values (Feature Scaling)
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())


# FEATURE ENGINEERING
# Creates location-based cluster similarity features from latitude & longitude
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)


# DATA CLEANING + FEATURE SCALING
# Default numeric pipeline for columns not listed explicitly
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())


# Complete PIPELINE — Combines all preprocessing steps
# Each tuple = ("name", transformer, columns)
preprocessing = ColumnTransformer([
        # Ratio features (Feature Engineering)
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        
        # Log transform features (Feature Transformation)
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        
        # Cluster similarity features (Feature Engineering)
        ("geo", cluster_simil, ["latitude", "longitude"]),
        
        # Categorical encoding (Feature Transformation)
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    # Default numeric handling for remaining columns (Data Cleaning + Scaling)
    remainder=default_num_pipeline)  
