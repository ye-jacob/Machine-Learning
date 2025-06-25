# Binary Classification - Build Your Own Algorithms
# A hands-on framework where YOU implement the core algorithms
# Solutions are provided in comments for learning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class BinaryClassificationFromScratch:
    """
    Framework for implementing binary classification algorithms from scratch.
    You'll build: Logistic Regression, Decision Tree, and k-NN
    """
    
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def generate_data(self, n_samples=300, n_features=2, noise=0.1):
        """Generate sample data for learning"""
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_redundant=0,
            n_informative=n_features, n_clusters_per_class=1, flip_y=noise, random_state=42
        )
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2):
        """Split data into train/test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
        
    def visualize_data(self, X, y, title="Dataset"):
        """Visualize 2D data"""
        if X.shape[1] == 2:
            plt.figure(figsize=(8, 6))
            colors = ['red', 'blue']
            for i, color in enumerate(colors):
                mask = y == i
                plt.scatter(X[mask, 0], X[mask, 1], c=color, label=f'Class {i}', alpha=0.7)
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

# =============================================================================
# EXERCISE 1: IMPLEMENT LOGISTIC REGRESSION FROM SCRATCH
# =============================================================================

class LogisticRegressionScratch:
    """
    YOUR TASK: Implement logistic regression from scratch!
    
    Key concepts to understand:
    - Sigmoid function: maps any real number to (0,1)
    - Cost function: Cross-entropy loss
    - Gradient descent: Iterative optimization
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def sigmoid(self, z):
        """
        YOUR TASK: Implement the sigmoid function
        Formula: σ(z) = 1 / (1 + e^(-z))
        
        Args:
            z: Linear combination (X @ weights + bias)
        Returns:
            Sigmoid output between 0 and 1
        """
        # TODO: Implement sigmoid function
        # Hint: Use np.exp() and handle overflow with np.clip()
        return 1 / (1 + np.exp(-z))
        
    
    def cost_function(self, y_true, y_pred):
        """
        YOUR TASK: Implement cross-entropy cost function
        Formula: J = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted probabilities
        Returns:
            Cost (scalar)
        """
        # TODO: Implement cross-entropy loss
        # Hint: Use np.log() and add small epsilon to prevent log(0)
        
        # SOLUTION (uncomment to see):
        m = len(y_true)
        epsilon = 1e-15  # Prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cost = -1/m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
        
        pass
    
    def fit(self, X, y):
        """
        YOUR TASK: Implement the training algorithm using gradient descent
        
        Steps:
        1. Initialize weights and bias
        2. For each iteration:
           a. Compute predictions using sigmoid
           b. Compute cost
           c. Compute gradients
           d. Update weights and bias
        """
        # TODO: Initialize weights and bias
        # Hint: weights shape should be (n_features,), bias is scalar
        
        # SOLUTION (uncomment to see):
        # n_samples, n_features = X.shape
        # self.weights = np.random.normal(0, 0.01, n_features)
        # self.bias = 0
        # 
        # for i in range(self.max_iterations):
        #     # Forward pass
        #     z = X @ self.weights + self.bias
        #     y_pred = self.sigmoid(z)
        #     
        #     # Compute cost
        #     cost = self.cost_function(y, y_pred)
        #     self.cost_history.append(cost)
        #     
        #     # Compute gradients
        #     dw = (1/n_samples) * X.T @ (y_pred - y)
        #     db = (1/n_samples) * np.sum(y_pred - y)
        #     
        #     # Update parameters
        #     self.weights -= self.learning_rate * dw
        #     self.bias -= self.learning_rate * db
        
        pass
    
    def predict(self, X):
        """
        YOUR TASK: Make predictions on new data
        
        Steps:
        1. Compute linear combination
        2. Apply sigmoid
        3. Convert probabilities to binary predictions (threshold = 0.5)
        """
        # TODO: Implement prediction
        
        # SOLUTION (uncomment to see):
        # z = X @ self.weights + self.bias
        # probabilities = self.sigmoid(z)
        # return (probabilities >= 0.5).astype(int)
        
        pass
    
    def predict_proba(self, X):
        """Return prediction probabilities"""
        # TODO: Return probabilities instead of binary predictions
        
        # SOLUTION (uncomment to see):
        # z = X @ self.weights + self.bias
        # return self.sigmoid(z)
        
        pass

# =============================================================================
# EXERCISE 2: IMPLEMENT DECISION TREE FROM SCRATCH
# =============================================================================

class DecisionTreeScratch:
    """
    YOUR TASK: Implement a simple decision tree for binary classification!
    
    Key concepts:
    - Information Gain: How much a split reduces uncertainty
    - Gini Impurity: Measure of node impurity
    - Recursive splitting: Build tree top-down
    """
    
    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def gini_impurity(self, y):
        """
        YOUR TASK: Calculate Gini impurity
        Formula: Gini = 1 - Σ(p_i^2) where p_i is proportion of class i
        
        Args:
            y: Labels for samples in this node
        Returns:
            Gini impurity (0 = pure, 0.5 = maximum impurity for binary)
        """
        # TODO: Implement Gini impurity
        # Hint: Use np.bincount() to count classes, then compute proportions
        
        # SOLUTION (uncomment to see):
        # if len(y) == 0:
        #     return 0
        # counts = np.bincount(y)
        # proportions = counts / len(y)
        # return 1 - np.sum(proportions ** 2)
        
        pass
    
    def information_gain(self, y_parent, y_left, y_right):
        """
        YOUR TASK: Calculate information gain from a split
        Formula: IG = Gini(parent) - [w_left * Gini(left) + w_right * Gini(right)]
        
        Args:
            y_parent: Labels before split
            y_left: Labels in left child
            y_right: Labels in right child
        Returns:
            Information gain
        """
        # TODO: Calculate weighted average of child impurities and subtract from parent
        
        # SOLUTION (uncomment to see):
        # n_parent = len(y_parent)
        # n_left, n_right = len(y_left), len(y_right)
        # 
        # if n_left == 0 or n_right == 0:
        #     return 0
        # 
        # gini_parent = self.gini_impurity(y_parent)
        # gini_left = self.gini_impurity(y_left)
        # gini_right = self.gini_impurity(y_right)
        # 
        # weighted_gini = (n_left/n_parent) * gini_left + (n_right/n_parent) * gini_right
        # return gini_parent - weighted_gini
        
        pass
    
    def find_best_split(self, X, y):
        """
        YOUR TASK: Find the best feature and threshold to split on
        
        Steps:
        1. Try each feature
        2. For each feature, try different thresholds
        3. Calculate information gain for each split
        4. Return the split with highest information gain
        """
        # TODO: Find best split by trying all features and thresholds
        
        # SOLUTION (uncomment to see):
        # best_gain = -1
        # best_feature = None
        # best_threshold = None
        # 
        # n_features = X.shape[1]
        # 
        # for feature in range(n_features):
        #     thresholds = np.unique(X[:, feature])
        #     
        #     for threshold in thresholds:
        #         # Split data
        #         left_mask = X[:, feature] <= threshold
        #         right_mask = ~left_mask
        #         
        #         if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
        #             continue
        #         
        #         # Calculate information gain
        #         gain = self.information_gain(y, y[left_mask], y[right_mask])
        #         
        #         if gain > best_gain:
        #             best_gain = gain
        #             best_feature = feature
        #             best_threshold = threshold
        # 
        # return best_feature, best_threshold, best_gain
        
        pass
    
    def build_tree(self, X, y, depth=0):
        """
        YOUR TASK: Recursively build the decision tree
        
        Stopping criteria:
        1. Maximum depth reached
        2. Too few samples to split
        3. All samples have same label (pure node)
        4. No information gain from any split
        """
        # TODO: Implement recursive tree building
        
        # SOLUTION (uncomment to see):
        # # Stopping criteria
        # if (depth >= self.max_depth or 
        #     len(y) < self.min_samples_split or 
        #     len(np.unique(y)) == 1):
        #     # Return leaf node with majority class
        #     return {'prediction': np.argmax(np.bincount(y))}
        # 
        # # Find best split
        # feature, threshold, gain = self.find_best_split(X, y)
        # 
        # if gain == 0:  # No useful split found
        #     return {'prediction': np.argmax(np.bincount(y))}
        # 
        # # Split data
        # left_mask = X[:, feature] <= threshold
        # right_mask = ~left_mask
        # 
        # # Recursively build children
        # left_child = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        # right_child = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        # 
        # return {
        #     'feature': feature,
        #     'threshold': threshold,
        #     'left': left_child,
        #     'right': right_child
        # }
        
        pass
    
    def fit(self, X, y):
        """Train the decision tree"""
        self.tree = self.build_tree(X, y)
    
    def predict_sample(self, sample, tree):
        """
        YOUR TASK: Predict a single sample using the tree
        
        Steps:
        1. Start at root
        2. Follow path based on feature values
        3. Return prediction when you reach a leaf
        """
        # TODO: Traverse tree to make prediction
        
        # SOLUTION (uncomment to see):
        # if 'prediction' in tree:  # Leaf node
        #     return tree['prediction']
        # 
        # if sample[tree['feature']] <= tree['threshold']:
        #     return self.predict_sample(sample, tree['left'])
        # else:
        #     return self.predict_sample(sample, tree['right'])
        
        pass
    
    def predict(self, X):
        """Predict multiple samples"""
        return np.array([self.predict_sample(sample, self.tree) for sample in X])

# =============================================================================
# EXERCISE 3: IMPLEMENT K-NEAREST NEIGHBORS FROM SCRATCH
# =============================================================================

class KNearestNeighborsScratch:
    """
    YOUR TASK: Implement k-NN classification!
    
    Key concepts:
    - Distance metrics (Euclidean)
    - Majority voting
    - No training phase (lazy learning)
    """
    
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def euclidean_distance(self, point1, point2):
        """
        YOUR TASK: Calculate Euclidean distance between two points
        Formula: d = √Σ(x_i - y_i)²
        """
        # TODO: Implement Euclidean distance
        
        # SOLUTION (uncomment to see):
        # return np.sqrt(np.sum((point1 - point2) ** 2))
        
        pass
    
    def fit(self, X, y):
        """
        YOUR TASK: 'Train' k-NN (just store the training data)
        k-NN is a lazy learner - no actual training happens!
        """
        # TODO: Store training data
        
        # SOLUTION (uncomment to see):
        # self.X_train = X
        # self.y_train = y
        
        pass
    
    def predict_single(self, sample):
        """
        YOUR TASK: Predict label for a single sample
        
        Steps:
        1. Calculate distance to all training samples
        2. Find k nearest neighbors
        3. Return majority vote among k neighbors
        """
        # TODO: Implement single sample prediction
        
        # SOLUTION (uncomment to see):
        # # Calculate distances to all training samples
        # distances = []
        # for train_sample in self.X_train:
        #     dist = self.euclidean_distance(sample, train_sample)
        #     distances.append(dist)
        # 
        # distances = np.array(distances)
        # 
        # # Find k nearest neighbors
        # k_nearest_indices = np.argsort(distances)[:self.k]
        # k_nearest_labels = self.y_train[k_nearest_indices]
        # 
        # # Return majority vote
        # return np.argmax(np.bincount(k_nearest_labels))
        
        pass
    
    def predict(self, X):
        """Predict labels for multiple samples"""
        return np.array([self.predict_single(sample) for sample in X])

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def calculate_accuracy(y_true, y_pred):
    """
    YOUR TASK: Calculate accuracy
    Formula: (True Positives + True Negatives) / Total
    """
    # TODO: Implement accuracy calculation
    
    # SOLUTION (uncomment to see):
    # return np.mean(y_true == y_pred)
    
    pass

def calculate_precision_recall(y_true, y_pred):
    """
    YOUR TASK: Calculate precision and recall
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    """
    # TODO: Calculate confusion matrix components and compute precision/recall
    
    # SOLUTION (uncomment to see):
    # tp = np.sum((y_true == 1) & (y_pred == 1))
    # fp = np.sum((y_true == 0) & (y_pred == 1))
    # fn = np.sum((y_true == 1) & (y_pred == 0))
    # 
    # precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # 
    # return precision, recall
    
    pass

def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    """Visualize decision boundary for 2D data"""
    if X.shape[1] != 2:
        print("Can only plot decision boundary for 2D data")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    
    # Plot data points
    colors = ['red', 'blue']
    for i, color in enumerate(colors):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=color, label=f'Class {i}', edgecolors='black')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.show()

# =============================================================================
# MAIN LEARNING WORKFLOW
# =============================================================================

def run_learning_workflow():
    """
    Complete workflow to test your implementations
    """
    print("🚀 BINARY CLASSIFICATION FROM SCRATCH")
    print("=" * 50)
    
    # Initialize framework
    framework = BinaryClassificationFromScratch()
    
    # Generate data
    print("📊 Generating sample data...")
    X, y = framework.generate_data(n_samples=300, n_features=2)
    framework.prepare_data(X, y)
    framework.visualize_data(X, y, "Original Dataset")
    
    print("\n" + "="*50)
    print("🎯 TESTING YOUR IMPLEMENTATIONS")
    print("="*50)
    
    # Test Logistic Regression
    print("\n1️⃣ Testing Logistic Regression...")
    lr = LogisticRegressionScratch(learning_rate=0.1, max_iterations=1000)
    try:
        lr.fit(framework.X_train, framework.y_train)
        lr_predictions = lr.predict(framework.X_test)
        lr_accuracy = calculate_accuracy(framework.y_test, lr_predictions)
        print(f"✅ Logistic Regression Accuracy: {lr_accuracy:.3f}")
        
        # Plot decision boundary
        plot_decision_boundary(framework.X_test, framework.y_test, lr, 
                             "Logistic Regression Decision Boundary")
        
        # Plot cost history
        if len(lr.cost_history) > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(lr.cost_history)
            plt.title('Logistic Regression - Cost vs Iterations')
            plt.xlabel('Iterations')
            plt.ylabel('Cost')
            plt.grid(True)
            plt.show()
            
    except Exception as e:
        print(f"❌ Logistic Regression failed: {e}")
        print("💡 Check your sigmoid, cost_function, and fit methods!")
    
    # Test Decision Tree
    print("\n2️⃣ Testing Decision Tree...")
    dt = DecisionTreeScratch(max_depth=5, min_samples_split=10)
    try:
        dt.fit(framework.X_train, framework.y_train)
        dt_predictions = dt.predict(framework.X_test)
        dt_accuracy = calculate_accuracy(framework.y_test, dt_predictions)
        print(f"✅ Decision Tree Accuracy: {dt_accuracy:.3f}")
        
        # Plot decision boundary
        plot_decision_boundary(framework.X_test, framework.y_test, dt, 
                             "Decision Tree Decision Boundary")
        
    except Exception as e:
        print(f"❌ Decision Tree failed: {e}")
        print("💡 Check your gini_impurity, find_best_split, and build_tree methods!")
    
    # Test k-NN
    print("\n3️⃣ Testing k-Nearest Neighbors...")
    knn = KNearestNeighborsScratch(k=5)
    try:
        knn.fit(framework.X_train, framework.y_train)
        knn_predictions = knn.predict(framework.X_test)
        knn_accuracy = calculate_accuracy(framework.y_test, knn_predictions)
        print(f"✅ k-NN Accuracy: {knn_accuracy:.3f}")
        
        # Plot decision boundary
        plot_decision_boundary(framework.X_test, framework.y_test, knn, 
                             "k-NN Decision Boundary")
        
    except Exception as e:
        print(f"❌ k-NN failed: {e}")
        print("💡 Check your euclidean_distance, fit, and predict_single methods!")
    
    print("\n" + "="*50)
    print("🎉 WORKFLOW COMPLETE!")
    print("="*50)
    
    return framework

# =============================================================================
# LEARNING EXERCISES
# =============================================================================

def learning_exercises():
    """
    Exercises to deepen your understanding
    """
    print("\n📚 LEARNING EXERCISES")
    print("=" * 30)
    
    exercises = [
        "1. 🧮 Mathematics Understanding:",
        "   • Derive the gradient of logistic regression cost function",
        "   • Understand why sigmoid function is used",
        "   • Research other impurity measures (entropy, classification error)",
        "",
        "2. 🔧 Implementation Improvements:",
        "   • Add regularization to logistic regression (L1/L2)",
        "   • Implement different distance metrics for k-NN (Manhattan, Cosine)",
        "   • Add pruning to decision tree to prevent overfitting",
        "",
        "3. 📊 Experimentation:",
        "   • Try different learning rates for logistic regression",
        "   • Test different k values for k-NN",
        "   • Vary max_depth for decision tree",
        "   • Compare performance on different datasets",
        "",
        "4. 🚀 Advanced Features:",
        "   • Implement feature normalization",
        "   • Add cross-validation",
        "   • Create ensemble methods (voting classifier)",
        "   • Handle missing values",
        "",
        "5. 🎯 Real-world Application:",
        "   • Load a real dataset (iris, wine, breast cancer)",
        "   • Compare your implementations with sklearn",
        "   • Analyze feature importance in decision tree",
        "   • Study the effect of class imbalance"
    ]
    
    for exercise in exercises:
        print(exercise)

if __name__ == "__main__":
    print("🎓 Welcome to Binary Classification from Scratch!")
    print("\n📝 YOUR MISSION:")
    print("Implement the core algorithms by filling in the TODO sections.")
    print("Solutions are provided in comments for learning.")
    print("\n🚀 When ready, run: run_learning_workflow()")
    print("\n💡 Start with uncommentng the SOLUTION comments to see how it works,")
    print("then try implementing it yourself!")
    
    # Uncomment the line below when you're ready to test your implementations:
    # run_learning_workflow()
    # learning_exercises()