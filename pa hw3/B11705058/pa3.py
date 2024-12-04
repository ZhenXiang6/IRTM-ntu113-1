import os
import math
import re
from collections import defaultdict, Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# 獲取腳本所在的目錄
script_dir = os.path.dirname(os.path.abspath(__file__))

# 設定 training.txt 和 test_docs 的路徑
training_file = os.path.join(script_dir, "training.txt")
test_folder = os.path.join(script_dir, "data")

# 嘗試下載停用詞資源，如果尚未下載
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# 初始化詞幹提取器
stemmer = PorterStemmer()

# 文本預處理函數
def preprocess(text):
    text = text.lower()  # 轉為小寫
    text = re.sub(r'[^a-z0-9\s]', '', text)  # 移除非字母數字字符
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # 去除停用詞並進行詞幹提取
    return ' '.join(words)

# 1. 讀取 training.txt 並加載訓練文檔內容
def load_training_data(training_file, test_folder):
    docs = []
    labels = []
    train_doc_ids = set()
    if not os.path.exists(training_file):
        print(f"訓練文件 {training_file} 不存在。請檢查路徑。")
        return docs, labels, train_doc_ids
    with open(training_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                print(f"格式錯誤的行：{line}")
                continue
            class_id = int(parts[0])  # 類別 ID
            doc_ids = map(int, parts[1:])  # 該類別的文檔 ID
            for doc_id in doc_ids:
                train_doc_ids.add(doc_id)
                file_path = os.path.join(test_folder, f"{doc_id}.txt")
                if not os.path.exists(file_path):
                    print(f"訓練文檔 {file_path} 未找到。")
                    continue
                try:
                    with open(file_path, 'r', encoding="utf-8") as file:
                        content = preprocess(file.read().strip())
                        docs.append(content)
                        labels.append(class_id)
                except Exception as e:
                    print(f"讀取訓練文檔 {file_path} 時出現錯誤：{e}")
    return docs, labels, train_doc_ids

# 2. 讀取 test 資料，排除已在訓練資料中的文檔
def load_test_data(test_folder, train_doc_ids):
    docs = []
    doc_ids = []
    if not os.path.exists(test_folder):
        print(f"測試資料夾 {test_folder} 不存在。請檢查路徑。")
        return docs, doc_ids
    for file_name in sorted(os.listdir(test_folder), key=lambda x: int(x.split(".")[0])):
        if file_name.endswith(".txt"):
            try:
                doc_id = int(file_name.split(".")[0])
            except ValueError:
                print(f"文件名 {file_name} 不是有效的文檔 ID。")
                continue
            if doc_id in train_doc_ids:
                continue  # 排除已在訓練資料中的文檔
            file_path = os.path.join(test_folder, file_name)
            if not os.path.exists(file_path):
                print(f"測試文檔 {file_path} 未找到。")
                continue
            try:
                with open(file_path, 'r', encoding="utf-8") as file:
                    content = preprocess(file.read().strip())
                    docs.append(content)
                    doc_ids.append(doc_id)
            except Exception as e:
                print(f"讀取測試文檔 {file_path} 時出現錯誤：{e}")
    return docs, doc_ids

# 3. 計算詞頻和文檔頻率
def compute_tf_df(docs):
    tf_list = []
    df = defaultdict(int)
    for doc in docs:
        terms = doc.split()
        tf = Counter(terms)
        tf_list.append(tf)
        for term in tf.keys():
            df[term] += 1
    return tf_list, df

# 4. 使用卡方檢驗選擇特徵
def chi_square_selection(docs, labels, df, num_features=500):
    class_term_count = defaultdict(lambda: defaultdict(int))
    class_doc_count = defaultdict(int)
    total_docs = len(docs)

    # 統計每個類別中的詞頻
    for doc, label in zip(docs, labels):
        terms = set(doc.split())
        class_doc_count[label] += 1
        for term in terms:
            class_term_count[label][term] += 1

    # 計算 Chi-Square 分數
    chi_scores = defaultdict(float)
    for term in df.keys():
        for class_id in class_doc_count.keys():
            A = class_term_count[class_id][term]
            B = sum(class_term_count[c][term] for c in class_doc_count if c != class_id)
            C = class_doc_count[class_id] - A
            D = total_docs - (A + B + C)

            numerator = (total_docs * (A * D - B * C) ** 2)
            denominator = (A + C) * (B + D) * (A + B) * (C + D)
            if denominator > 0:
                chi_scores[term] += numerator / denominator

    # 選擇最高分的詞
    selected_features = sorted(chi_scores, key=chi_scores.get, reverse=True)[:num_features]
    return selected_features

# 5. 訓練 Naive Bayes classifier
def train_naive_bayes(docs, labels, selected_features):
    class_term_count = defaultdict(lambda: defaultdict(int))
    class_doc_count = defaultdict(int)
    vocab_size = len(selected_features)
    total_docs = len(docs)

    # 計算每個類別的詞頻與文檔數
    for doc, label in zip(docs, labels):
        terms = doc.split()
        class_doc_count[label] += 1
        for term in terms:
            if term in selected_features:
                class_term_count[label][term] += 1

    # 計算先驗機率
    class_priors = {c: math.log(count / total_docs) for c, count in class_doc_count.items()}

    # 計算條件機率並取對數
    class_word_probs = defaultdict(dict)
    for class_id, term_count in class_term_count.items():
        total_terms = sum(term_count.values())
        for term in selected_features:
            # 使用拉普拉斯平滑並取對數
            class_word_probs[class_id][term] = math.log((term_count[term] + 1) / (total_terms + vocab_size))

    return class_priors, class_word_probs

# 6. 預測函數
def predict_naive_bayes(docs, class_priors, class_word_probs, selected_features):
    predictions = []
    for doc in docs:
        terms = doc.split()
        class_scores = {}
        for class_id in class_priors:
            score = class_priors[class_id]
            for term in terms:
                if term in selected_features:
                    score += class_word_probs[class_id].get(term, math.log(1 / (1 + len(selected_features))))
            class_scores[class_id] = score
        if class_scores:
            predictions.append(max(class_scores, key=class_scores.get))
        else:
            # 若無法計算分數，則預設為類別 1（或其他邏輯）
            predictions.append(1)
    return predictions

# 7. 主函數
if __name__ == "__main__":
    # 1. 載入訓練資料
    print("載入訓練資料...")
    train_docs, train_labels, train_doc_ids = load_training_data(training_file, test_folder)
    print(f"訓練文檔數量：{len(train_docs)}")

    # 2. 載入測試資料，排除訓練文檔
    print("載入測試資料，並排除訓練文檔...")
    test_docs, test_doc_ids = load_test_data(test_folder, train_doc_ids)
    print(f"測試文檔數量：{len(test_docs)}")

    # 3. 計算詞頻與文檔頻率
    print("計算詞頻與文檔頻率...")
    tf_list, df = compute_tf_df(train_docs)

    # 4. 特徵選擇
    print("進行卡方特徵選擇...")
    selected_features = chi_square_selection(train_docs, train_labels, df, num_features=500)
    print(f"選擇了 {len(selected_features)} 個特徵詞")

    # 5. 訓練模型
    print("訓練 Naive Bayes classifier...")
    class_priors, class_word_probs = train_naive_bayes(train_docs, train_labels, selected_features)

    # 6. 預測
    print("對測試文檔進行預測...")
    predictions = predict_naive_bayes(test_docs, class_priors, class_word_probs, selected_features)

    # 7. 輸出結果
    print("輸出預測結果到 submission.csv...")
    submission_file = os.path.join(script_dir, "submission.csv")
    with open(submission_file, "w", encoding="utf-8") as f:
        f.write("Id,Class\n")  # 修改列名為 "Id,Class" 或根據需要調整
        for doc_id, pred in zip(test_doc_ids, predictions):
            f.write(f"{doc_id},{pred}\n")
    print("Results saved to submission.csv")
