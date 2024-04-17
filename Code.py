import tensorflow as tf
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

class SimilarityCalculator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def encode_sentences(self, sentences):
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        return embeddings
    
    def cosine_similarity_normalized(self, embeddings1, embeddings2):
        dot_product = tf.reduce_sum(embeddings1 * embeddings2, axis=1)
        norm_embeddings1 = tf.norm(embeddings1, axis=1)
        norm_embeddings2 = tf.norm(embeddings2, axis=1)
        cosine_similarity = dot_product / (norm_embeddings1 * norm_embeddings2)
        normalized_similarity = (cosine_similarity + 1) / 2
        return normalized_similarity.numpy()
    
    def compute_normalized_similarity(self, text1_list, text2_list):
        embeddings1 = self.encode_sentences(text1_list)
        embeddings2 = self.encode_sentences(text2_list)
        normalized_similarity = self.cosine_similarity_normalized(embeddings1, embeddings2)
        return normalized_similarity

similarity_calculator = SimilarityCalculator()

@app.route('/', methods=['GET'])
def index():
    return "Welcome to Sentence Similarity Calculator made by Mohd Sarique for DataNeuron!"

@app.route('/similarity', methods=['POST'])
def compute_similarity():
    data = request.get_json()
    text1_list = data.get('text1_list', [])
    text2_list = data.get('text2_list', [])
    
    if not text1_list or not text2_list:
        return jsonify({'error': 'Please provide text1_list and text2_list in the request body.'}), 400
    
    normalized_similarity = similarity_calculator.compute_normalized_similarity(text1_list, text2_list)
    
    return jsonify({'normalized_similarity': normalized_similarity.tolist()}), 200

if __name__ == '__main__':
    app.run(debug=True)

