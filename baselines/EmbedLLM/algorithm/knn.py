from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import argparse
import pandas as pd
import torch
import numpy as np

def process_csv_data(data_path):
    """
    Convert CSV data to tensor format with embeddings
    (This would take a while so we suggest to save the tensor data in the first run)
    """
    print(f"Processing CSV data from {data_path}...")
    df = pd.read_csv(data_path)
    
    df = df.groupby(['model_id', 'prompt_id', 'prompt']).agg({'label': 'max'}).reset_index()
    
    # Generate question embeddings
    embedder = SentenceTransformer('all-mpnet-base-v2')
    unique_questions = df[['prompt_id', 'prompt']].drop_duplicates()
    unique_questions['embedding'] = unique_questions['prompt'].apply(lambda x: embedder.encode(x, show_progress_bar=True))
    
    # Merge embeddings back
    df = df.merge(unique_questions[['prompt_id', 'embedding']], on='prompt_id')
    
    # Create correctness matrix
    correctness_matrix = df.pivot(index='model_id', columns='prompt_id', values='label').fillna(0).astype(int)
    correctness_array = correctness_matrix.values
    
    # Create final tensor
    question_embeddings = np.stack(unique_questions.set_index('prompt_id').loc[correctness_matrix.columns]['embedding'].values)
    final_tensor = np.array([question_embeddings] * correctness_array.shape[0])
    
    # Split into features and labels
    tensor_x = torch.tensor(final_tensor)
    tensor_y = torch.tensor(correctness_array)
    
    print(f"Processed data shapes - X: {tensor_x.shape}, y: {tensor_y.shape}")
    return tensor_x, tensor_y

def save_tensor_data(tensor_x, tensor_y, save_x_path, save_y_path):
    """Save processed tensors to files"""
    torch.save(tensor_x, save_x_path)
    torch.save(tensor_y, save_y_path)
    print(f"Tensors saved to {save_x_path} and {save_y_path}")

def load_tensor_data(train_x_path, train_y_path, test_x_path, test_y_path):
    """Load data from pre-processed tensor files"""
    train_x = torch.load(train_x_path)
    train_y = torch.load(train_y_path)
    test_x = torch.load(test_x_path)
    test_y = torch.load(test_y_path)

    print(f"Data shapes - train_x: {train_x.shape}, train_y: {train_y.shape}, test_x: {test_x.shape}, test_y: {test_y.shape}")
    return train_x, train_y, test_x, test_y

def load_csv_data(train_csv_path, test_csv_path):
    """Load and process data from CSV files"""
    train_x, train_y = process_csv_data(train_csv_path)
    test_x, test_y = process_csv_data(test_csv_path)
    
    print("Final shapes:")
    print(f"Train - X: {train_x.shape}, y: {train_y.shape}")
    print(f"Test - X: {test_x.shape}, y: {test_y.shape}")
    return train_x, train_y, test_x, test_y

def evaluate(train_x, train_y, test_x, test_y, num_neighbors):
    accs = []
    num_model, q_embed_dim, num_train_questions = train_x.shape
    for i in tqdm(range(num_model)):
        X_train, y_train = train_x[i, :, :].tolist(), train_y[i, :].tolist()
        X_test, y_test = test_x[i, :, :].tolist(), test_y[i, :].tolist()
        
        neigh = KNeighborsClassifier(n_neighbors=num_neighbors)
        neigh.fit(X_train, y_train)
        
        y_pred = neigh.predict(X_test).tolist()
        bool_ls = list(map(lambda x, y: int(x == y), y_pred, y_test))
        accs.append(sum(bool_ls) / len(y_test))

    mean_accuracy = sum(accs) / len(accs)
    print(f"Mean Test Accuracy for {num_neighbors} neighbors: {mean_accuracy:.4f}")
    return {"mean_accuracy": mean_accuracy}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-format", type=str, choices=['tensor', 'csv'], default='csv',
                       help="Format of input data: 'tensor' for .pth files or 'csv' for raw data")
    parser.add_argument("--train-x-path", type=str, default='../data/train_x.pth')
    parser.add_argument("--train-y-path", type=str, default='../data/train_y.pth')
    parser.add_argument("--test-x-path", type=str, default='../data/test_x.pth')
    parser.add_argument("--test-y-path", type=str, default='../data/test_y.pth')
    parser.add_argument("--train-csv-path", type=str, default='../data/train.csv')
    parser.add_argument("--test-csv-path", type=str, default='../data/test.csv')
    parser.add_argument("--num-neighbors", type=int, default=131) # We found 131 to be the best number of neighbors from validation set
    parser.add_argument("--save-tensors", action='store_true', 
                        help="Save processed CSV data as tensors") # We suggest to save the tensors to avoid re-processing the data
    parser.add_argument("--save-train-x-path", type=str, default='../data/train_x.pth')
    parser.add_argument("--save-train-y-path", type=str, default='../data/train_y.pth')
    parser.add_argument("--save-test-x-path", type=str, default='../data/test_x.pth')
    parser.add_argument("--save-test-y-path", type=str, default='../data/test_y.pth')
    args = parser.parse_args()

    print("Start Initializing Dataset...")
    if args.input_format == 'tensor':
        train_x, train_y, test_x, test_y = load_tensor_data(
            args.train_x_path, args.train_y_path, args.test_x_path, args.test_y_path)
    else:
        train_x, train_y, test_x, test_y = load_csv_data(args.train_csv, args.test_csv)
        if args.save_tensors:
            print("Saving tensors...")
            save_tensor_data(train_x, train_y, args.save_train_x, args.save_train_y)
            save_tensor_data(test_x, test_y, args.save_test_x, args.save_test_y)
    print("Finish Initializing Dataset")
    
    evaluate(train_x, train_y, test_x, test_y, args.num_neighbors)