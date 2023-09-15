from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
import json


def jaccard_similarity(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    if len(set1.union(set2)) == 0:
        return 0
    return len(set1.intersection(set2)) / float(len(set1.union(set2)))

def load_jsonl(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]

def calculate_cosine_similarity(tfidf_vectorizer, str1, str2):
    tfidf_matrix = tfidf_vectorizer.transform([str1, str2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

if __name__ == "__main__":
    file1 = '/home/ec2-user/environment/data_generation/analysis/llm_training_data_0.jsonl'
    file2 = '/home/ec2-user/environment/data_generation/analysis/llm_training_data_n.jsonl'

    data1 = load_jsonl(file1)
    data2 = load_jsonl(file2)

    tfidf_vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the text data
    tfidf_vectorizer.fit([d['content'] for convo in data1 for d in convo] + 
                         [d['content'] for convo in data2 for d in convo])

    cos_sim_list = []
    jac_sim_list = []

    with open("detailed_metrics.txt", "w") as f:
        for i, (convo1, convo2) in enumerate(zip(data1, data2)):
            f.write(f"Comparing line {i + 1}...\n")
            
            for role_content1, role_content2 in zip(convo1, convo2):
                content1 = role_content1['content']
                content2 = role_content2['content']
                
                cos_sim = calculate_cosine_similarity(tfidf_vectorizer, content1, content2)
                jac_sim = jaccard_similarity(content1, content2)
                
                cos_sim_list.append(cos_sim)
                jac_sim_list.append(jac_sim)
                
                f.write(f"Cosine Similarity: {cos_sim}\n")
                f.write(f"Jaccard Similarity: {jac_sim}\n")
            f.write("------\n")
    
    avg_cos_sim = sum(cos_sim_list) / len(cos_sim_list)
    avg_jac_sim = sum(jac_sim_list) / len(jac_sim_list)

    print(f"Average Cosine Similarity: {avg_cos_sim}")
    print(f"Average Jaccard Similarity: {avg_jac_sim}")
