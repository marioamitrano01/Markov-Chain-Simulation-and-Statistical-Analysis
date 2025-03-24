import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm, skew, kurtosis, shapiro
import time
from functools import wraps
from sklearn.linear_model import LogisticRegression
import pandas as pd

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"Function {func.__name__} executed in {elapsed:.4f} seconds")
        return result
    return wrapper

class MarkovChain:
    """
    This class defines a finite-state Markov chain simulation.
    """
    def __init__(self, transition_matrix, state_space, initial_state, random_seed=None):
        self.transition_matrix = np.array(transition_matrix)
        self.state_space = state_space
        self.current_state = initial_state
        if random_seed is not None:
            np.random.seed(random_seed)
    def simulate(self, n_steps):
        states = [self.current_state]
        for _ in range(n_steps):
            current_index = self.state_space.index(self.current_state)
            self.current_state = np.random.choice(self.state_space, p=self.transition_matrix[current_index])
            states.append(self.current_state)
        return states
    def reset(self, state):
        self.current_state = state
    def compute_stationary_distribution(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        stationary = np.real(eigenvectors[:, np.isclose(eigenvalues, 1)])
        stationary = stationary[:, 0]
        stationary = stationary / stationary.sum()
        return dict(zip(self.state_space, stationary))

class SimulationExperiment:
    """
    This class runs multiple experiments on a given Markov chain and analyzes the distribution of sample means.
    """
    def __init__(self, markov_chain, reward_function, chain_length, num_experiments):
        self.markov_chain = markov_chain
        self.reward_function = reward_function
        self.chain_length = chain_length
        self.num_experiments = num_experiments
        self.sample_means = []
    @timing_decorator
    def run_experiments(self):
        initial_state = self.markov_chain.current_state
        for _ in range(self.num_experiments):
            self.markov_chain.reset(initial_state)
            states = self.markov_chain.simulate(self.chain_length)
            rewards = list(map(self.reward_function, states))
            self.sample_means.append(np.mean(rewards))
        return self.sample_means
    def compute_statistics(self):
        means = np.array(self.sample_means)
        mean_val = np.mean(means)
        std_val = np.std(means)
        skew_val = skew(means)
        kurt_val = kurtosis(means)
        stat_test = shapiro(means)
        return {"mean": mean_val, "std": std_val, "skew": skew_val, "kurtosis": kurt_val, "shapiro_stat": stat_test[0], "shapiro_p": stat_test[1]}
    def bootstrap_confidence_interval(self, num_bootstrap=1000, alpha=0.05):
        means = np.array(self.sample_means)
        boot_means = []
        n = len(means)
        for _ in range(num_bootstrap):
            sample = np.random.choice(means, size=n, replace=True)
            boot_means.append(np.mean(sample))
        lower = np.percentile(boot_means, 100*alpha/2)
        upper = np.percentile(boot_means, 100*(1-alpha/2))
        return lower, upper
    def qq_plot(self):
        means = np.array(self.sample_means)
        sorted_means = np.sort(means)
        n = len(sorted_means)
        probs = (np.arange(1, n+1) - 0.5) / n
        theoretical_quantiles = norm.ppf(probs, loc=np.mean(means), scale=np.std(means))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sorted_means, mode='markers', name='Empirical Quantiles'))
        min_val = min(np.min(theoretical_quantiles), np.min(sorted_means))
        max_val = max(np.max(theoretical_quantiles), np.max(sorted_means))
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='45-degree Line'))
        fig.update_layout(title="Q-Q Plot of Sample Means vs Normal Distribution",
                          xaxis_title="Theoretical Quantiles",
                          yaxis_title="Empirical Quantiles")
        fig.show()
    def kde_plot(self):
        means = np.array(self.sample_means)
        hist_values, bin_edges = np.histogram(means, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bin_centers, y=hist_values, name="Empirical Density", opacity=0.6))
        x_values = np.linspace(np.min(means), np.max(means), 200)
        normal_pdf = norm.pdf(x_values, loc=np.mean(means), scale=np.std(means))
        fig.add_trace(go.Scatter(x=x_values, y=normal_pdf, mode='lines', name="Theoretical Normal PDF"))
        fig.update_layout(title="Kernel Density Estimate of Sample Means with Normal PDF Overlay",
                          xaxis_title="Sample Means",
                          yaxis_title="Density")
        fig.show()

class MLTransitionEstimator:
    """
    This class applies a machine learning approach (multinomial logistic regression) to estimate the transition matrix of a Markov chain.
    """
    def __init__(self, state_space, model=None):
        self.state_space = state_space
        self.model = model if model is not None else LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
        self.fitted = False
    def prepare_dataset(self, state_sequence):
        current_states = state_sequence[:-1]
        next_states = state_sequence[1:]
        df = pd.DataFrame({'current': current_states, 'next': next_states})
        df['current_cat'] = df['current'].astype('category')
        X = pd.get_dummies(df['current_cat'])
        y = df['next'].astype('category').cat.codes
        self.category_mapping = dict(enumerate(df['next'].astype('category').cat.categories))
        return X, y
    @timing_decorator
    def fit(self, state_sequence):
        X, y = self.prepare_dataset(state_sequence)
        self.model.fit(X, y)
        self.fitted = True
        return self
    def predict_proba_for_state(self, state):
        if not self.fitted:
            raise ValueError("Model is not fitted yet")
        dummy_df = pd.DataFrame({'current': [state]})
        dummy_df['current_cat'] = dummy_df['current'].astype('category')
        dummy_df = pd.get_dummies(dummy_df['current_cat'])
        all_columns = pd.get_dummies(pd.Categorical(self.state_space, categories=self.state_space))
        dummy_df = dummy_df.reindex(columns=all_columns.columns, fill_value=0)
        proba = self.model.predict_proba(dummy_df)
        proba_dict = {self.state_space[i]: proba[0][i] for i in range(len(self.state_space))}
        return proba_dict
    def estimate_transition_matrix(self):
        estimated_matrix = []
        for state in self.state_space:
            proba_dict = self.predict_proba_for_state(state)
            estimated_matrix.append([proba_dict[s] for s in self.state_space])
        return np.array(estimated_matrix)
    def plot_transition_matrix_difference(self, true_matrix):
        estimated_matrix = self.estimate_transition_matrix()
        diff_matrix = np.array(true_matrix) - estimated_matrix
        fig = go.Figure(data=go.Heatmap(z=diff_matrix, x=self.state_space, y=self.state_space, colorscale='RdBu', colorbar=dict(title="Difference")))
        fig.update_layout(title="Difference between True and Estimated Transition Matrices",
                          xaxis_title="Next State",
                          yaxis_title="Current State")
        fig.show()

def reward_function(state):
    return state

def main():
    state_space = [1, 2, 3, 4]
    transition_matrix = [
        [0.1, 0.4, 0.4, 0.1],
        [0.2, 0.2, 0.5, 0.1],
        [0.3, 0.3, 0.3, 0.1],
        [0.25, 0.25, 0.25, 0.25]
    ]
    initial_state = 1
    mc = MarkovChain(transition_matrix, state_space, initial_state, random_seed=42)
    stationary_distribution = mc.compute_stationary_distribution()
    print("Stationary Distribution:", stationary_distribution)
    long_sequence = mc.simulate(10000)
    ml_estimator = MLTransitionEstimator(state_space)
    ml_estimator.fit(long_sequence)
    ml_estimator.plot_transition_matrix_difference(transition_matrix)
    experiment = SimulationExperiment(mc, reward_function, chain_length=1000, num_experiments=500)
    sample_means = experiment.run_experiments()
    stats = experiment.compute_statistics()
    print("Statistics of Sample Means:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    ci_lower, ci_upper = experiment.bootstrap_confidence_interval(num_bootstrap=1000, alpha=0.05)
    print(f"Bootstrap 95% Confidence Interval for the Mean: [{ci_lower}, {ci_upper}]")
    experiment.kde_plot()
    experiment.qq_plot()

if __name__ == "__main__":
    main()
