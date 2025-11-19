#  Quora Question Pair Similarity Classification

This project aims to build a machine learning model to classify pairs of questions from the Quora platform as either **duplicate** (semantically similar) or **not duplicate**. The solution involves comprehensive Exploratory Data Analysis (EDA), robust text preprocessing, feature engineering, and binary classification using a high-performing model.

## 🎯 Goal

To predict the target variable `is_duplicate` (1 for duplicate, 0 for not duplicate) based on the text of `question1` and `question2`.

---

## 🚀 Step 1: Exploratory Data Analysis (EDA)

The EDA phase focused on understanding the data structure, identifying class imbalance, and extracting initial predictive features based on question text length.

### 1. Class Imbalance Check



* **Analysis:** The distribution plot shows that approximately **63%** of the pairs are **Not Duplicate** (0), while only **37%** are **Duplicate** (1).
* **Impact:** Due to this class imbalance, we must use **F1-Score** and **AUC-ROC** as primary evaluation metrics, as simple accuracy would be misleading. For Model training, we have to go trough over sampling or under sampling for better model performane

<p align="center">
  <img src="assests/eda_duplicate_distribution.png" alt="Target Variable Distribution" width="500"/>
</p>

### 2. Length Feature Analysis (Word Count Difference)



* **Feature:** The `word_count_diff` feature was engineered to measure the absolute difference in word count between the two questions in a pair.
* **Analysis:** The KDE plot shows that the distribution for **Duplicate** pairs (blue curve) is highly concentrated around **zero** difference. This is a critical finding, confirming that pairs of questions with very similar lengths are much more likely to be true duplicates.

<p align="center">
  <img src="assests/eda_word_count_diff_distribution.png" alt="Word Count Difference Distribution" width="500"/>
</p>

### 3. Sentence Length Distribution

This plot analyzes the distribution of the **word count** for both Question 1 and Question 2 after initial cleaning.

* **Key Insight:** The density curves for $Q1$ and $Q2$ overlap almost perfectly, with the peak around 8–12 words. This confirms that questions in the dataset are **structurally similar in length** across the board. This basic characteristic needs to be augmented by more semantic features to distinguish duplicates.


<p align="center">
  <img src="assests/eda_sentence_length_distribution.png" alt="Sentence Length Distribution (Word Counts)" width="500"/>
</p>

### 4. Common Words Count vs. Duplication

This box plot compares the number of words shared between the question pairs, directly against the target variable.

* **Feature:** This plot visualizes the predictive power of the **`common_words`** feature (count of intersecting words).
* **Key Insight:** The median and overall distribution for the **Duplicate (1)** group is shifted significantly **higher** than the Non-Duplicate (0) group. This is a critical finding: pairs that are true duplicates share a far greater number of words. The `common_words` count is, therefore, a **highly effective feature** for the classification model.


<p align="center">
  <img src="assests/eda_common_words_boxplot.png" alt="Common Words Count Box Plot" width="500"/>
</p>

### 5. Word Cloud Comparison

These visualizations show the most frequent words (excluding common English stopwords) used in the questions, separated by the duplication status.

#### **Word Cloud for Duplicate Pairs**

* **Key Insight:** This cloud confirms that duplicate questions frequently cluster around **specific, narrow topics** (e.g., product names, popular technologies, financial terms). The highly prominent words here serve as strong indicators of where the core duplication issues lie.

<p align="center">
  <img src="assests/eda_wordcloud_duplicates.png" alt="Word Cloud for Duplicate Pairs" width="500"/>
</p>

#### **Word Cloud for Non-Duplicate Pairs**

* **Key Insight:** Words in this cloud are generally **more generic or broad**. Comparing the two clouds shows that questions marked as non-duplicate have a wider variety of themes, lacking the intense frequency clustering seen in the duplicate set.


<p align="center">
  <img src="assests/eda_wordcloud_non_duplicates.png" alt="Word Cloud for Non-Duplicate Pairs" width="500"/>
</p>

### 6. Feature Correlation Matrix Analysis

This visualization, a **Heatmap**, quantifies the linear relationship between the simple numerical features (length and count) and the target variable, `is_duplicate`.


<p align="center">
  <img src="assests/eda_correlation_matrix.png" alt="Correlation Matrix Heatmap" width="500"/>
</p>

* **What Correlation Means:** Values range from $+1.0$ (perfect positive relationship) to $-1.0$ (perfect negative relationship). A value close to $0$ means little to no linear relationship.

| Relationship | Correlation Coefficient | Interpretation |
| :--- | :--- | :--- |
| **`word_count_diff` vs. `is_duplicate`** | **$-0.20$** | **The most significant signal.** This indicates a weak but important **negative correlation**. As the absolute difference in word count increases (i.e., questions become more structurally dissimilar), the probability of the pair being a duplicate decreases. |
| **Individual Lengths/Counts vs. `is_duplicate`** | $\approx -0.15$ to $-0.17$ | The individual length/word count of $Q1$ or $Q2$ has a very weak relationship with duplication, confirming that the **difference** between the two questions is a much stronger indicator than their individual sizes. |
| **Internal Feature Correlation** | $\approx +0.95$ to $+0.97$ | Notice the high correlation between `q1_len` and `q1_word_count` (and similarly for $Q2$). This is expected: more characters generally mean more words. This confirms our features are consistent. |

**Conclusion:** The $-0.20$ correlation from `word_count_diff` confirms our hypothesis from the KDE plot: **structural similarity is predictive of duplication**, and this engineered feature is the most valuable simple numerical feature we can add to our text model.


---

## ⚙️ Step 2: Text Preprocessing and Feature Engineering

This step transformed the raw text into a high-dimensional feature matrix suitable for machine learning, while also managing class imbalance.

### 1. Preprocessing Pipeline

The cleaned text underwent the following normalization process:

* **Standardization:** Lowercasing, removal of special characters, punctuation, and English stopwords.
* **Normalization:** Applied **Lemmatization** (using NLTK's WordNetLemmatizer) to reduce words to their base or root form (e.g., "running" becomes "run"), which helps the model treat different word forms as the same feature.

### 2. Feature Extraction and Matrix Creation

We used a combined approach for feature extraction:

1.  **Vectorization (TF-IDF):** Questions were vectorized using **Term Frequency-Inverse Document Frequency** (TF-IDF), limited to the top 20,000 most frequent words.
2. **Feature Stacking:** The final feature matrix was constructed by combining four distinct sets of features:
 
 * **Absolute Difference Vector** ($\mathbf{|TFIDF_{Q1} - TFIDF_{Q2}|}$): This captures the **dissimilarity** between the questions.
 * **Element-wise Product Vector** ($\mathbf{TFIDF_{Q1} \cdot TFIDF_{Q2}}$): This captures the **overlap** between the questions.
 * **`common_words`** (Engineered numerical feature): A direct count of shared words.
 * **`word_count_diff`** (Engineered numerical feature): The absolute difference in question length.

* **Final Feature Matrix Shape:** `(404351, 40002)`
    * *Explanation:* The matrix has 404,351 rows (original data points) and **40,002 columns** (features). This comes from (20,000 difference features + 20,000 product features + 2 numerical features).

### 3. Data Splitting and Stratification

The data was split into Training, Validation, and Testing sets using stratification to ensure the duplicate ratio ($\approx 37\%$) is maintained across all three sets.

| Set | Proportion | Size (Rows) | Purpose |
| :--- | :--- | :--- | :--- |
| **Training** | 70% | 283,045 | Model learning and fitting. |
| **Validation** | 10% | 40,435 | Hyperparameter tuning and model selection. |
| **Testing** | 20% | 80,871 | Final, unbiased evaluation of the best model. |

### 4. Class Imbalance Correction (Oversampling)

To prevent the model from being biased toward the majority class (Non-Duplicates), **Random OverSampling (ROS)** was applied *only* to the training set.

* **Original Training Duplicates (1):** 104,514
* **Resampled Training Duplicates (1):** 178,531
* **Resampled Training Total Size:** 357,062 rows

The training data is now perfectly balanced (50/50), which is essential for achieving reliable **Precision** and **Recall** metrics.
