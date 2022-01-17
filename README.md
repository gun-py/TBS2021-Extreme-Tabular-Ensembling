# TBS2021-Extreme-Tabular-Ensembling
Gold Region Solution on TBS 2021_1.0 datasets

Tree-based models are used in the ensemble and they are LightGBM, CatBoost, XGBoost, and Random Forest. All of them are trained with more than 3 seeds for diversity. Only raw continuous features are used in tree-based models without any transformation. Linear models are used in the ensemble and they are Ridge Regression and SVM. Ridge is fitted to very high dimensional sparse features extracted by RandomTreesEmbedding, but can't be used on SVM because cuml. SVM doesn't support scipy.sparse.csr_matrix format along with Neural Nets mode from scratch are used. 

Dataset Link: https://drive.google.com/drive/folders/1U51kHU1m2JdhoC7hQbWmGlRHoRztUPl1?usp=sharing
