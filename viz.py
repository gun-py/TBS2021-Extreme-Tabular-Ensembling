df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

continuous_features = [feature for feature in df_train.columns if feature.startswith('cont')]
target = 'target'

print(f'Training Set Shape = {df_train.shape}')
print(f'Training Set Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
print(f'Test Set Shape = {df_test.shape}')
print(f'Test Set Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')


def plot_target(target):   
    print(f'Target feature {target} Statistical Analysis\n{"-" * 42}')   
    print(f'Mean: {df_train[target].mean():.4}  -  Median: {df_train[target].median():.4}  -  Std: {df_train[target].std():.4}')
    print(f'Min: {df_train[target].min():.4}  -  25%: {df_train[target].quantile(0.25):.4}  -  50%: {df_train[target].quantile(0.5):.4}  -  75%: {df_train[target].quantile(0.75):.4}  -  Max: {df_train[target].max():.4}')
    print(f'Skew: {df_train[target].skew():.4}  -  Kurtosis: {df_train[target].kurtosis():.4}')
    missing_values_count = df_train[df_train[target].isnull()].shape[0]
    training_samples_count = df_train.shape[0]
    print(f'Missing Values: {missing_values_count}/{training_samples_count} ({missing_values_count * 100 / training_samples_count:.4}%)')
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 12), dpi=100)
    sns.distplot(df_train[target], label=target, ax=axes[0][0])
    axes[0][0].axvline(df_train[target].mean(), label='Target Mean', color='r', linewidth=2, linestyle='--')
    axes[0][0].axvline(df_train[target].median(), label='Target Median', color='b', linewidth=2, linestyle='--')
    probplot(df_train[target], plot=axes[0][1])
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(df_train[target].values.reshape(-1, 1))
    df_train[f'{target}_class'] = gmm.predict(df_train[target].values.reshape(-1, 1)) 
    sns.distplot(df_train[target], label=target, ax=axes[1][0])
    sns.distplot(df_train[df_train[f'{target}_class'] == 0][target], label='Component 1', ax=axes[1][1])
    sns.distplot(df_train[df_train[f'{target}_class'] == 1][target], label='Component 2', ax=axes[1][1]) 
    axes[0][0].legend(prop={'size': 15})
    axes[1][1].legend(prop={'size': 15})
    for i in range(2):
        for j in range(2):
            axes[i][j].tick_params(axis='x', labelsize=12)
            axes[i][j].tick_params(axis='y', labelsize=12)
            axes[i][j].set_xlabel('')
            axes[i][j].set_ylabel('')
    axes[0][0].set_title(f'{target} Distribution in Training Set', fontsize=15, pad=12)
    axes[0][1].set_title(f'{target} Probability Plot', fontsize=15, pad=12)
    axes[1][0].set_title(f'{target} Distribution Before GMM', fontsize=15, pad=12)
    axes[1][1].set_title(f'{target} Distribution After GMM', fontsize=15, pad=12)
    plt.show()
    
plot_target(target)

def plot_continuous(continuous_feature):
    print(f'Continuous feature {continuous_feature} Statistical Analysis\n{"-" * 42}')
    print(f'Training Mean: {float(df_train[continuous_feature].mean()):.4}  - Training Median: {float(df_train[continuous_feature].median()):.4} - Training Std: {float(df_train[continuous_feature].std()):.4}')
    print(f'Test Mean: {float(df_test[continuous_feature].mean()):.4}  - Test Median: {float(df_test[continuous_feature].median()):.4} - Test Std: {float(df_test[continuous_feature].std()):.4}')
    print(f'Training Min: {float(df_train[continuous_feature].min()):.4}  - Training Max: {float(df_train[continuous_feature].max()):.4}')
    print(f'Test Min: {float(df_test[continuous_feature].min()):.4}  - Training Max: {float(df_test[continuous_feature].max()):.4}')
    print(f'Training Skew: {float(df_train[continuous_feature].skew()):.4}  - Training Kurtosis: {float(df_train[continuous_feature].kurtosis()):.4}')
    print(f'Test Skew: {float(df_test[continuous_feature].skew()):.4}  - Test Kurtosis: {float(df_test[continuous_feature].kurtosis()):.4}')
    training_missing_values_count = df_train[df_train[continuous_feature].isnull()].shape[0]
    test_missing_values_count = df_test[df_test[continuous_feature].isnull()].shape[0]
    training_samples_count = df_train.shape[0]
    test_samples_count = df_test.shape[0]
    print(f'Training Missing Values: {training_missing_values_count}/{training_samples_count} ({training_missing_values_count * 100 / training_samples_count:.4}%)')
    print(f'Test Missing Values: {test_missing_values_count}/{test_samples_count} ({test_missing_values_count * 100 / test_samples_count:.4}%)')
    fig, axes = plt.subplots(ncols=2, figsize=(24, 6), dpi=100, constrained_layout=True)
    title_size = 18
    label_size = 18
    # Continuous Feature Training and Test Set Distribution
    sns.distplot(df_train[continuous_feature], label='Training', ax=axes[0])
    sns.distplot(df_test[continuous_feature], label='Test', ax=axes[0])
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', labelsize=label_size)
    axes[0].tick_params(axis='y', labelsize=label_size)
    axes[0].legend()
    axes[0].set_title(f'{continuous_feature} Distribution in Training and Test Set', size=title_size, pad=title_size)
    # Continuous Feature vs target
    sns.scatterplot(df_train[continuous_feature], df_train[target], ax=axes[1])
    axes[1].set_title(f'{continuous_feature} vs {target}', size=title_size, pad=title_size)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].tick_params(axis='x', labelsize=label_size)
    axes[1].tick_params(axis='y', labelsize=label_size) 
    plt.show()
    
for continuous_feature in sorted(continuous_features, key=lambda x: int(x.split('cont')[-1])):
    plot_continuous(continuous_feature)