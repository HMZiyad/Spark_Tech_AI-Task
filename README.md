#  Quora Question Pair Similarity Classification

This project aims to build a machine learning model to classify pairs of questions from the Quora platform as either **duplicate** (semantically similar) or **not duplicate**. The solution involves comprehensive Exploratory Data Analysis (EDA), robust text preprocessing, feature engineering, and binary classification using a high-performing model.

## 🎯 Goal

To predict the target variable `is_duplicate` (1 for duplicate, 0 for not duplicate) based on the text of `question1` and `question2`.

---

## 🚀 Step 1: Exploratory Data Analysis (EDA)

The EDA phase focused on understanding the data structure, identifying class imbalance, and extracting initial predictive features based on question text length.

### 1. Class Imbalance Check



* **Analysis:** The distribution plot shows that approximately **63%** of the pairs are **Not Duplicate** (0), while only **37%** are **Duplicate** (1).
* **Impact:** Due to this class imbalance, we must use **F1-Score** and **AUC-ROC** as primary evaluation metrics, as simple accuracy would be misleading.

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
  <img src="assests/eda_wordcloud_duplicates.jpg" alt="Word Cloud for Duplicate Pairs" width="500"/>
</p>

#### **Word Cloud for Non-Duplicate Pairs**

* **Key Insight:** Words in this cloud are generally **more generic or broad**. Comparing the two clouds shows that questions marked as non-duplicate have a wider variety of themes, lacking the intense frequency clustering seen in the duplicate set.

![Word Cloud for Non-Duplicate Pairs](assets/eda_wordcloud_non_duplicates.jpg)
<p align="center">
  <img src="assests/eda_wordcloud_non_duplicates.jpg" alt="Word Cloud for Non-Duplicate Pairs" width="500"/>
</p>

### 6. Feature Correlation



* **Analysis:** The correlation heatmap was used to quantify the relationship between our engineered length features and the target variable.
    * The **`word_count_diff`** showed the strongest correlation with `is_duplicate` at **-0.20**. This negative value confirms that as the difference in word count increases, the probability of the pair being a duplicate decreases.
* **Conclusion:** The `word_count_diff` feature will be included in the final model as a powerful numerical feature alongside vectorized text.

<p align="center">
  <img src="assests/eda_correlation_matrix.png" alt="Correlation Matrix Heatmap" width="500"/>
</p>
---

## ⚙️ Step 2: Text Preprocessing and Feature Engineering

This step converts raw text into a numerical format suitable for machine learning.

1.  **Text Cleaning:**
    * Converted all text to **lowercase**.
    * Handled missing values by replacing NaNs with an empty string.
    * Removed HTML tags (e.g., `[math]...[/math]`).
    * Removed punctuation and special characters.
    * Removed **English stopwords** (e.g., 'the', 'a', 'is').
2.  **Feature Augmentation:** Created the following features based on the *cleaned* text:
    * **`common_words`:** Count of common words between the two questions.
    * **`word_count_diff`:** (Re-calculated on cleaned text) Absolute difference in word count.
3.  **Vectorization (TF-IDF):**
    * Used **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert the cleaned questions into sparse numerical vectors.
    * The final feature matrix combines four components for maximum predictive power:
        * $|\text{TFIDF}_{\text{Q1}} - \text{TFIDF}_{\text{Q2}}|$ (Absolute Difference)
        * $\text{TFIDF}_{\text{Q1}} \cdot \text{TFIDF}_{\text{Q2}}$ (Element-wise Product)
        * $\text{common\_words}$ (Numerical Feature)
        * $\text{word\_count\_diff}$ (Numerical Feature)
4.  **Data Split:** The final feature matrix was split into 80% training data and 20% test data, using **stratification** to preserve the class balance ratio in both sets.

---

## 🧠 Step 3: Model Building and Evaluation

We established a strong baseline using **Logistic Regression** and then recommend an advanced model for production.

### Baseline Model: Logistic Regression

| Metric | Result (Example) | Justification |
| :--- | :--- | :--- |
| **Accuracy** | 76.50% | Base measure, but misleading due to imbalance. |
| **F1-Score** | 68.00% | **Primary Metric:** Best balance between Precision and Recall. |
| **Recall** | 60.50% | Measures how many duplicates we correctly identified. |
| **Precision** | 77.00% | Measures how many of our positive predictions were correct. |
| **AUC-ROC** | 83.50% | Measures the model's ability to discriminate between the two classes. |

### Confusion Matrix

| | Predicted 0 | Predicted 1 |
| :--- | :--- | :--- |
| **Actual 0** | (True Negatives) | (False Positives) |
| **Actual 1** | (False Negatives) | (True Positives) |

### Optimization & Future Work

* **Model Tuning:** The next step is to perform **Grid Search** or **Random Search** on the Logistic Regression hyperparameters (e.g., the `C` parameter) to maximize the F1-Score.
* **Advanced Models:** Experimenting with gradient boosting machines like **XGBoost** or **LightGBM** is highly recommended, as they typically outperform linear models on complex, large-scale feature sets.
