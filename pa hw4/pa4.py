import re
import math
import numpy as np
from os import listdir
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
from collections import defaultdict
import sys
import os
import heapq

# 確保已經下載了 NLTK 的停用詞
nltk.download('stopwords')

# --------------------------- #
# 1. 設置全局變量
# --------------------------- #

# Stemmer
STEMMER = PorterStemmer()

# Stop words
STOP_WORDS = set(stopwords.words("english"))

# Corpus file path (data folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_FILE_PATH = os.path.join(SCRIPT_DIR, "data")

# Doc Size
DOC_SIZE = 1095  # 根據實際文檔數量調整

# --------------------------- #
# 2. 文檔前處理函數
# --------------------------- #

def doc_preprocessing(doc: str) -> list:
    """
    清理文本、分詞、詞幹提取和移除停用詞。
    """
    # 清理文本並轉換為小寫
    doc = re.sub(r"\s+", " ", doc)
    doc = re.sub(r"[^\w\s]", "", doc)
    doc = doc.lower()
    # 分詞
    words = doc.split(" ")
    # 詞幹提取
    stemming = [STEMMER.stem(word) for word in words]
    # 移除停用詞和空字符串
    token_list = [word for word in stemming if word and word not in STOP_WORDS]
    return token_list

# --------------------------- #
# 3. 計算 TF 與 DF 的函數
# --------------------------- #

def get_tf_and_df(corpus: list):
    """
    計算每個文檔的詞頻（TF）和語料庫的文件頻率（DF）。
    """
    tf_list = []
    df_dict = defaultdict(int)
    
    for document_id, document in corpus:
        document_word_list = doc_preprocessing(document)
        tf = defaultdict(int)
        for word in document_word_list:
            tf[word] += 1
        tf_list.append([document_id, dict(tf)])
        
        for word in tf.keys():
            df_dict[word] += 1
                
    # 按照詞彙排序 DF 字典
    df_dict = dict(sorted(df_dict.items()))
    
    return tf_list, df_dict

# --------------------------- #
# 4. 建立索引字典函數
# --------------------------- #

def get_index_dict(df_dict: dict) -> dict:
    """
    為每個詞彙分配一個唯一的索引。
    """
    index_dict = {term: idx for idx, term in enumerate(df_dict)}
    return index_dict  # (word: index)

# --------------------------- #
# 5. 計算 TF 向量與 TF-IDF 向量的函數
# --------------------------- #

def get_tf_vector(tf_list, index_dict):
    """
    將每個文檔的詞頻轉換為向量表示。
    """
    tf_vectors = []
    for document_id, tf_dict in tf_list:
        tf_vector = np.zeros(len(index_dict), dtype=float)
        for word, count in tf_dict.items():
            if word in index_dict:  # 確保詞彙在索引字典中
                tf_vector[index_dict[word]] = count
        tf_vectors.append(tf_vector)
    return np.array(tf_vectors)

def get_tf_idf_vector(tf_vectors, df_dict, index_dict):
    """
    計算 TF-IDF 向量並進行歸一化。
    """
    N = len(tf_vectors)
    idf_vector = np.log(N / np.array([df_dict[word] for word in index_dict.keys()]))
    
    tf_idf_vectors = tf_vectors * idf_vector
    # 歸一化
    norms = np.linalg.norm(tf_idf_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 防止除以零
    tf_idf_unit = tf_idf_vectors / norms
    return tf_idf_unit

# --------------------------- #
# 6. 定義餘弦相似度函數
# --------------------------- #

def cosine(doc_x, doc_y, doc_vectors):
    """
    計算兩個文檔之間的餘弦相似度。
    """
    return float(np.dot(doc_vectors[doc_x], doc_vectors[doc_y]))

# --------------------------- #
# 7. 定義 MaxHeap 類
# --------------------------- #

class MaxHeap:
    def __init__(self):
        self.heap = []
    
    def push(self, similarity, cluster1, cluster2):
        """
        將新的相似度和聚類對插入堆中。
        使用負相似度來模擬最大堆。
        """
        heapq.heappush(self.heap, (-similarity, cluster1, cluster2))
    
    def extract_max(self):
        """
        提取堆中相似度最高的聚類對。
        """
        if not self.heap:
            return None
        sim, cluster1, cluster2 = heapq.heappop(self.heap)
        return (-sim, cluster1, cluster2)
    
    def is_empty(self):
        return len(self.heap) == 0

# --------------------------- #
# 8. 加載並處理文檔
# --------------------------- #

def load_documents(corpus_path, doc_size):
    """
    從指定目錄中讀取所有文檔，並存儲到語料庫列表中。
    """
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus path '{corpus_path}' does not exist.")
        print(f"請確認資料夾位於：{corpus_path}")
        sys.exit(1)
    
    # 加載文檔
    files = listdir(corpus_path)
    files = [f for f in files if f.endswith(".txt") and not f.startswith(".")]

    # 按照文件名排序（假設文件名為數字，如1.txt, 2.txt, ...）
    try:
        files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    except IndexError as ie:
        print(f"Error sorting files: {ie}")
        sys.exit(1)
    except ValueError as ve:
        print(f"Error parsing file names: {ve}")
        sys.exit(1)

    # 初始化語料庫列表：[[id, document], ...]
    corpus = []

    # 讀取文件
    for file in files[:doc_size]:
        file_path = os.path.join(corpus_path, file)
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                document = f.read()
                # 提取數字作為ID，從0開始
                document_id = int(re.findall(r'\d+', file)[0]) - 1
                corpus.append([document_id, document])
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except ValueError as ve:
            print(f"Error processing file {file}: {ve}")
        except Exception as e:
            print(f"Unexpected error processing file {file}: {e}")

    print(f"Loaded {len(corpus)} documents.")
    return corpus

# --------------------------- #
# 9. 聚類過程
# --------------------------- #

def write_result(hac_dict, cluster_num):
    """
    將聚類結果寫入文件，並在終端顯示。
    """
    file_path = os.path.join(SCRIPT_DIR, f"{cluster_num}.txt")
    try:
        with open(file_path, "w") as f:
            for cluster_id, docs in hac_dict.items():
                for doc_id in sorted(docs):
                    f.write(f"{doc_id + 1}\n")
                f.write("\n")
        print(f"聚類數量 {cluster_num} 的結果文件已生成：{file_path}")
    except Exception as e:
        print(f"Error writing result to file {file_path}: {e}")

def hac_complete_linkage(corpus, tf_idf_vectors, target_clusters):
    """
    實現 Complete-Linkage HAC，使用 MaxHeap 作為優先隊列。
    """
    DOC_SIZE = len(corpus)
    hac_dict = {i: [i] for i in range(DOC_SIZE)}
    active_clusters = set(range(DOC_SIZE))

    # 初始化 MaxHeap
    heap = MaxHeap()

    # 計算初始相似度並插入堆中
    print("Initializing heap with all document similarities...")
    for i in range(DOC_SIZE):
        if (i + 1) % 100 == 0 or i == DOC_SIZE - 1:
            print(f"Processed {i + 1} / {DOC_SIZE} documents for heap initialization...")
        for j in range(i + 1, DOC_SIZE):
            sim = cosine(i, j, tf_idf_vectors)
            heap.push(sim, i, j)
    
    print("Heap initialization complete.")
    print(f"Heap size: {len(heap.heap)}")

    # 聚類過程
    print("Starting clustering process...")
    while len(hac_dict) > min(target_clusters):
        if heap.is_empty():
            print("Heap is empty. Ending clustering.")
            break
        max_sim, cluster1, cluster2 = heap.extract_max()
        if cluster1 not in active_clusters or cluster2 not in active_clusters:
            # 此聚類對中至少有一個已被合併，跳過
            continue  # 跳過無效的聚類對

        # 合併聚類
        hac_dict[cluster1].extend(hac_dict[cluster2])
        del hac_dict[cluster2]
        active_clusters.remove(cluster2)
        print(f"Merged cluster {cluster2} into cluster {cluster1}. Current number of clusters: {len(hac_dict)}")

        # 更新相似度堆
        for other in active_clusters:
            if other == cluster1:
                continue
            # 計算新的相似度（全鏈法：取最小相似度）
            sim1 = cosine(cluster1, other, tf_idf_vectors)
            sim2 = cosine(cluster2, other, tf_idf_vectors)
            new_sim = min(sim1, sim2)
            heap.push(new_sim, cluster1, other)
            # Debug信息
            # print(f"Updated similarity between {cluster1} and {other}: {new_sim:.4f}")

        # 檢查是否達到目標聚類數量
        current_cluster_num = len(hac_dict)
        if current_cluster_num in target_clusters:
            print(f"Writing result for {current_cluster_num} clusters...")
            write_result(hac_dict, current_cluster_num)

        # 定期打印進度
        if current_cluster_num % 100 == 0 or current_cluster_num in target_clusters:
            print(f"Current number of clusters: {current_cluster_num}")

    print("Clustering complete.")
    return hac_dict

# --------------------------- #
# 10. 主程序
# --------------------------- #

def main():
    # 確認資料夾存在
    if not os.path.exists(CORPUS_FILE_PATH):
        print(f"Error: Corpus path '{CORPUS_FILE_PATH}' does not exist.")
        print(f"請確認資料夾位於：{CORPUS_FILE_PATH}")
        sys.exit(1)
    
    # 加載文檔
    corpus = load_documents(CORPUS_FILE_PATH, DOC_SIZE)

    # 計算 TF 和 DF
    tf_list, df_dict = get_tf_and_df(corpus)
    print(f"Calculated TF for {len(tf_list)} documents.")
    print(f"Vocabulary size: {len(df_dict)}")

    # 建立索引字典
    index_dict = get_index_dict(df_dict)
    print(f"Index dictionary created with {len(index_dict)} terms.")

    # 生成 TF 向量
    tf_vectors = get_tf_vector(tf_list, index_dict)
    print(f"Generated TF vectors for {tf_vectors.shape[0]} documents.")

    # 生成 TF-IDF 向量
    tf_idf_vectors = get_tf_idf_vector(tf_vectors, df_dict, index_dict)
    doc_vectors = tf_idf_vectors  # 已經是 2D NumPy 數組
    print("TF-IDF vectors generated and normalized.")
    print(f"doc_vectors shape: {doc_vectors.shape}")

    # 定義目標聚類數量
    target_clusters = [20, 13, 8]

    # 執行 HAC
    hac_complete_linkage(corpus, doc_vectors, target_clusters)

    # 檢查結果文件是否存在
    for num in target_clusters:
        file_path = os.path.join(SCRIPT_DIR, f"{num}.txt")
        if os.path.exists(file_path):
            print(f"聚類數量 {num} 的結果文件已生成：{file_path}")
        else:
            print(f"聚類數量 {num} 的結果文件不存在。")

if __name__ == "__main__":
    main()
