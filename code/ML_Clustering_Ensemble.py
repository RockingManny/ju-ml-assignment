import math
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, mean_squared_error
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
import time
from tqdm import tqdm

class ClusteringComparison:
    def __init__(self, dataset_name, k_values=[], m=5, algorithm=None):
        self.dataset_name = dataset_name
        self.k_values = k_values
        self.m = m  # Number of different clustering models
        self.algorithm = algorithm
        # self.data = self.load_data()
        # self.true_labels = None  # Placeholder for unsupervised
        self.data, self.true_labels = self.load_data()
        self.scaled_data = self.preprocess_data()

        self.results_folder = os.path.join("OUTPUT", f"{dataset_name}_dataset")
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
        
        # Create a subfolder for visualizations
        self.visualizations_folder = os.path.join(self.results_folder, "visualizations")
        if not os.path.exists(self.visualizations_folder):
            os.makedirs(self.visualizations_folder)

    def load_data(self):
        print(f"\n[INFO] Loading dataset: {self.dataset_name}...")
        if self.dataset_name == "mnist":
            mnist = fetch_openml('mnist_784', version=1, as_frame=True)

            print("\n[INFO] MNIST dataset loaded successfully!")
            # Print data
            print(pd.DataFrame(mnist.data))

            self.m = random.randint(math.ceil(len(mnist.data.columns) // 2), len(mnist.data.columns))
            print(f"Number of features: {len(mnist.data.columns)}")
            print(f"Number of features to select: {self.m}")

            return mnist.data, mnist.target.astype(int)  # Ensure labels are integer
        elif self.dataset_name == "diabetes":
            url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
            col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]
            print("\n[INFO] Diabetes dataset loaded successfully!")
            data = pd.read_csv(url, names=col_names)

            # Print data
            print(data)
            self.m = random.randint(math.ceil(len(data.columns) // 2), len(data.columns))
            print(f"Number of features: {len(data.columns)}")
            print(f"Number of features to select: {self.m}")

            return data.iloc[:, :-1], data.iloc[:, -1]  # Features, Labels
        else:
            raise ValueError("Dataset not supported!")
    
    def save_plot(self, fig, filename):
        """Helper function to save plots in the visualizations folder."""
        filepath = os.path.join(self.visualizations_folder, filename)
        # make folders if they don't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(filepath)
        print(f"Plot saved to: {filepath}")
    
    def preprocess_data(self):
        print("[INFO] Preprocessing data (Scaling) + PCA for MNIST)...")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)

        # Apply PCA for MNIST (reducing to 50 components)
        if self.dataset_name == "mnist":
            print("[INFO] Preprocessing data (PCA for MNIST)...")
            pca = PCA(n_components=50)
            scaled_data = pca.fit_transform(scaled_data)
        
        return scaled_data
    
    def cluster_with_feature_subsets(self, k):
        print(f"[INFO] Performing feature-subset clustering for K={k}...")
        n_features = self.scaled_data.shape[1]
        cluster_assignments = np.zeros((self.scaled_data.shape[0], self.m * k))
        
        for i in range(self.m):
            selected_features = np.random.choice(range(n_features), size=math.floor(n_features ** 0.5), replace=False)
            sub_data = self.scaled_data[:, selected_features]
            
            if not self.algorithm or self.algorithm not in ["kmeans", "dbscan"]:
                # randomly select between KMeans and DBSCAN
                self.algorithm = random.choice(["kmeans", "dbscan"])

            if self.algorithm == "dbscan":
                model = DBSCAN(eps=1.5, min_samples=5).fit(sub_data)
            else:
                model = KMeans(n_clusters=k, random_state=i).fit(sub_data)
            
            self.visualize_clusters(model, k, additional=f"subcluster_{i}_{self.algorithm}")

            for j in range(k):
                cluster_assignments[:, i * k + j] = (model.labels_ == j).astype(int)
        
        return cluster_assignments
    
    def final_kmeans_clustering(self, transformed_data, k):
        print(f"[INFO] Applying final K-Means clustering on transformed data for K={k}...")
        model = KMeans(n_clusters=k, random_state=42).fit(transformed_data)
        return model.labels_
    
    def evaluate_clustering(self, true_labels, pred_labels):
        rand_idx = adjusted_rand_score(true_labels, pred_labels)
        mse = mean_squared_error(true_labels, pred_labels)
        entropy_val = entropy(np.bincount(pred_labels))
        return {"Rand Index": rand_idx, "MSE": mse, "Entropy": entropy_val}
    
    def run_experiment(self):
        results = []
        
        # Generate random lower and upper bounds
        if not self.k_values:
            print("Generating random K values...")
            lower_bound = random.randint(math.floor(len(self.data.columns)*(2/7)), len(self.data.columns) // 2)
            print(f"Lower bound: {lower_bound}")
            upper_bound = random.randint(lower_bound, math.floor(len(self.data.columns)*(5/7)))
            print(f"Upper bound: {upper_bound}")

            n = random.randint(math.ceil(len(self.data.columns) // 2), len(self.data.columns))
            print(f"Number of K values: {n}")
            # Number of elements in the list

            # Generate the random list within the bounds
            self.k_values = [random.randint(lower_bound, upper_bound) for i in range(n)]
            print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
        print(self.k_values)


        total_k = len(self.k_values)
        start_time = time.time()
        
        with tqdm(total=total_k, desc="Clustering Progress", unit="K") as pbar:
            for k in self.k_values:
                iter_start = time.time()

                print(f"\n[INFO] Running clustering experiment for K={k}...")
                transformed_data = self.cluster_with_feature_subsets(k)


                final_labels = self.final_kmeans_clustering(transformed_data, k)

                
                traditional_kmeans_model = KMeans(n_clusters=k, random_state=42).fit(self.scaled_data)
                traditional_labels = traditional_kmeans_model.labels_

                
                eval_ensemble = self.evaluate_clustering(self.true_labels, final_labels)
                eval_traditional = self.evaluate_clustering(self.true_labels, traditional_labels)

                
                results.append({
                    "K": k,
                    "Ensemble Rand Index": eval_ensemble["Rand Index"],
                    "Traditional Rand Index": eval_traditional["Rand Index"],
                    "Ensemble MSE": eval_ensemble["MSE"],
                    "Traditional MSE": eval_traditional["MSE"],
                    "Ensemble Entropy": eval_ensemble["Entropy"],
                    "Traditional Entropy": eval_traditional["Entropy"]
                })

                self.visualize_clusters(traditional_kmeans_model, k, additional="traditional_kmeans")

                elapsed_time = time.time() - start_time
                avg_time_per_iter = elapsed_time / (pbar.n + 1)
                expected_total_time = avg_time_per_iter * total_k
                remaining_time = expected_total_time - elapsed_time
                
                print(f"Elapsed: {elapsed_time:.2f}s, Remaining: {remaining_time:.2f}s")
                pbar.update(1)
                
                
        df_results = pd.DataFrame(results)
        print("\n[INFO] Clustering Experiment Completed!")
        print(df_results.to_string(index=False))
        self.visualize_results(df_results)
        return df_results
    
    def visualize_clusters(self, model, k, additional=''):
        print(f"[INFO] Visualizing clusters for K={k}...")
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.scaled_data)
        labels = model.labels_
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
        plt.colorbar(scatter)
        plt.title(f"Cluster Visualization (K={k})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        # plt.show()
        # target_path = os.path.join("visualizations", f"{additional}_clusters_k{k}.png")
        self.save_plot(plt, f"{additional}_clusters_k{k}.png")
        plt.close()
    
    def visualize_results(self, results_df):
        metrics = ["Rand Index", "MSE", "Entropy"]
        for metric in metrics:
            plt.figure(figsize=(10, 5))
            sns.lineplot(x=results_df["K"], y=results_df[f"Ensemble {metric}"], marker='o', label="Ensemble")
            sns.lineplot(x=results_df["K"], y=results_df[f"Traditional {metric}"], marker='s', label="Traditional")
            plt.xlabel("K")
            plt.ylabel(metric)
            plt.title(f"Comparison of {metric}")
            plt.legend()
            # plt.show()
            # target_path = os.path.join("visualizations", f"{metric}_comparison.png")
            self.save_plot(plt, f"{metric}_comparison.png")
            plt.close()

# Example usage
# # mnist_experiment = ClusteringComparison("mnist", k_values=[10, 11, 12, 13,14,15])
# mnist_experiment = ClusteringComparison("mnist")
# mnist_results = mnist_experiment.run_experiment()

# diabetes_experiment = ClusteringComparison("diabetes", k_values=[2, 3, 4, 5])
diabetes_experiment = ClusteringComparison("diabetes")
diabetes_results = diabetes_experiment.run_experiment()
