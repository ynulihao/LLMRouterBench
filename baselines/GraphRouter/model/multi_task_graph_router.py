import random
import numpy as np
import torch
from pathlib import Path
from graph_nn import  form_data,GNN_prediction
from data_processing.utils import savejson,loadjson,savepkl,loadpkl
import pandas as pd
import json
import re
import yaml
device = "cuda" if torch.cuda.is_available() else "cpu"

class graph_router_prediction:
    def __init__(self, router_data_path,llm_path,llm_embedding_path,config,wandb):
        self.config = config
        self.wandb = wandb
        self.data_df = self._load_router_data(router_data_path)
        self.llm_description = loadjson(llm_path)
        self.llm_names = list(self.llm_description.keys())
        self.num_llms=len(self.llm_names)
        self.num_query=int(len(self.data_df)/self.num_llms)
        # num_task is no longer needed - inferred from data_df['task_id'] in split_data()
        self.set_seed(self.config['seed'])
        self.llm_description_embedding=loadpkl(llm_embedding_path)
        self.prepare_data_for_GNN()
        self.split_data()
        self.form_data = form_data(device)
        self.query_dim = self.query_embedding_list.shape[1]
        self.llm_dim = self.llm_description_embedding.shape[1]
        self.GNN_predict = GNN_prediction(query_feature_dim=self.query_dim, llm_feature_dim=self.llm_dim,
                                    hidden_features_size=self.config['embedding_dim'], in_edges_size=self.config['edge_dim'],wandb=self.wandb,config=self.config,device=device)
        print("GNN training successfully initialized.")
        self.train_GNN()


    def set_seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_router_data(self, router_data_path):
        """
        Load router data with automatic format detection.

        Supports both CSV and Parquet formats for backward compatibility.
        Parquet format is preferred for better performance (3x faster I/O, 80% smaller file size).

        Args:
            router_data_path: Path to router data file (can be .csv or .parquet, or without extension)

        Returns:
            pandas.DataFrame with router data

        Raises:
            FileNotFoundError: If neither CSV nor Parquet file exists
        """
        path = Path(router_data_path)

        # If path has no extension, try to detect format
        if path.suffix == '':
            parquet_path = path.with_suffix('.parquet')
            csv_path = path.with_suffix('.csv')
        elif path.suffix == '.parquet':
            parquet_path = path
            csv_path = path.with_suffix('.csv')
        else:  # .csv or other
            csv_path = path
            parquet_path = path.with_suffix('.parquet')

        # Try Parquet first (preferred format)
        if parquet_path.exists():
            print(f"Loading router data from Parquet: {parquet_path}")
            return pd.read_parquet(parquet_path)

        # Fallback to CSV for backward compatibility
        elif csv_path.exists():
            print(f"Loading router data from CSV: {csv_path}")
            print("Note: Consider using Parquet format for 3x faster loading. Run adaptor to generate .parquet file.")
            return pd.read_csv(csv_path)

        # Neither format exists
        else:
            raise FileNotFoundError(
                f"Router data not found. Tried:\n"
                f"  - {parquet_path} (recommended)\n"
                f"  - {csv_path} (legacy)\n"
                f"Please run the GraphRouter adaptor to generate data files."
            )

    def split_data(self):
        """
        Split data into train/test sets based on task_id column.

        Modified for LLMRouterBench integration to properly handle datasets with
        varying sample sizes. Each task (dataset) is split independently using
        the configured split_ratio, preventing data leakage across tasks.

        IMPORTANT: Data is split by query (not by row). Each query has
        self.num_llms rows (one per model), and all rows for a query must be
        in the same split to ensure proper reshaping during training.

        NOTE: split_ratio[1] is set to 0.0 (no validation set). The validation
        mask is kept for code compatibility but will be all zeros.
        """
        from collections import defaultdict

        split_ratio = self.config['split_ratio']

        # Group rows by (task_id, query) to identify unique queries
        query_groups = defaultdict(list)
        for idx, row in self.data_df.iterrows():
            query_key = (row['task_id'], row['query'])
            query_groups[query_key].append(idx)

        # Group queries by task_id
        task_queries = defaultdict(list)
        for (task_id, query), row_indices in query_groups.items():
            task_queries[task_id].append(row_indices)

        # Split each task's queries independently
        train_idx = []
        validate_idx = []
        test_idx = []

        for task_id, queries in sorted(task_queries.items()):
            n_queries = len(queries)
            train_size = int(n_queries * split_ratio[0])
            val_size = int(n_queries * split_ratio[1])
            # test_size is the remainder

            # Split queries (each query is a list of row indices for all models)
            for i, query_rows in enumerate(queries):
                if i < train_size:
                    train_idx.extend(query_rows)
                elif i < train_size + val_size:
                    validate_idx.extend(query_rows)
                else:
                    test_idx.extend(query_rows)


        # Preserve the raw effect (task score) separately for evaluation metrics
        # combined_edge columns before concatenation in form_data: [0]=cost, [1]=raw_effect (score)
        self.combined_edge = np.concatenate((self.cost_list.reshape(-1, 1),
                                             self.effect_list.reshape(-1, 1)), axis=1)

        # Compute the routing utility that mixes performance and cost per scenario.
        # NOTE: We switch training labels to follow the upstream implementation:
        #   - edge_attr (self.effect_list) becomes the utility per edge
        #   - labels are one-hot on the LLM with maximum utility for each query
        # Evaluation (task success rate and cost) still uses raw_effect and cost_usd
        # from combined_edge, as implemented in graph_nn.py.
        self.scenario = self.config['scenario']
        if self.scenario == "Performance First":
            self.effect_list = 1.0 * self.effect_list - 0.0 * self.cost_list
        elif self.scenario == "Balance":
            self.effect_list = 0.5 * self.effect_list - 0.5 * self.cost_list
        else:
            self.effect_list = 0.2 * self.effect_list - 0.8 * self.cost_list

        # PATCH: Multi-label targets for ties — set all max-utility LLMs to 1
        # This keeps BCE-compatible labels while avoiding arbitrary tie-breaking.
        utility_matrix = self.effect_list.reshape(-1, self.num_llms)
        tie_atol = self.config.get('tie_atol', 1e-8)
        row_max = np.max(utility_matrix, axis=1, keepdims=True)
        tie_mask = np.isclose(utility_matrix, row_max, atol=tie_atol)
        self.label = tie_mask.astype(np.float32).reshape(-1, 1)
        self.edge_org_id=[num for num in range(self.num_query) for _ in range(self.num_llms)]
        self.edge_des_id=list(range(self.edge_org_id[0],self.edge_org_id[0]+self.num_llms))*self.num_query

        self.mask_train =torch.zeros(len(self.edge_org_id))
        self.mask_train[train_idx]=1

        self.mask_validate = torch.zeros(len(self.edge_org_id))
        self.mask_validate[validate_idx] = 1

        self.mask_test = torch.zeros(len(self.edge_org_id))
        self.mask_test[test_idx] = 1


    def prepare_data_for_GNN(self):
        unique_index_list=list(range(0, len(self.data_df), self.num_llms))
        query_embedding_list_raw=self.data_df['query_embedding'].tolist()
        task_embedding_list_raw = self.data_df['task_description_embedding'].tolist()
        self.query_embedding_list= []
        self.task_embedding_list= []
        for inter in query_embedding_list_raw:
            inter=re.sub(r'\s+', ', ', inter.strip())
            try:
                inter=json.loads(inter)
            except:
                inter = inter.replace("[[,", "[[")
                inter = json.loads(inter)
            self.query_embedding_list.append(inter[0])

        for inter in task_embedding_list_raw:
            inter=re.sub(r'\s+', ', ', inter.strip())
            try:
                inter=json.loads(inter)
            except:
                inter = inter.replace("[[,", "[[")
                inter = json.loads(inter)
            self.task_embedding_list.append(inter[0])
        self.query_embedding_list=np.array(self.query_embedding_list)[unique_index_list]
        self.task_embedding_list = np.array(self.task_embedding_list)[unique_index_list]
        self.effect_list=np.array(self.data_df['effect'].tolist())
        self.cost_list=np.array(self.data_df['cost'].tolist())
        # Preserve USD costs for evaluation reporting if available
        if 'cost_usd' in self.data_df.columns:
            self.cost_usd_list = np.array(self.data_df['cost_usd'].tolist())
        else:
            # Fallback to normalized cost if USD not present
            self.cost_usd_list = self.cost_list.copy()

        # Extract task_id for each query (for per-task evaluation)
        self.query_task_ids = self.data_df['task_id'].iloc[unique_index_list].tolist()



    def train_GNN(self):

        self.data_for_GNN_train = self.form_data.formulation(task_id=self.task_embedding_list,
                                                             query_feature=self.query_embedding_list,
                                                             llm_feature=self.llm_description_embedding,
                                                             org_node=self.edge_org_id,
                                                             des_node=self.edge_des_id,
                                                             edge_feature=self.effect_list, edge_mask=self.mask_train,
                                                             label=self.label, combined_edge=self.combined_edge,
                                                             train_mask=self.mask_train, valide_mask=self.mask_validate,
                                                             test_mask=self.mask_test, cost_usd=self.cost_usd_list)
        self.data_for_GNN_validate = self.form_data.formulation(task_id=self.task_embedding_list,
                                                                query_feature=self.query_embedding_list,
                                                                llm_feature=self.llm_description_embedding,
                                                                org_node=self.edge_org_id,
                                                                des_node=self.edge_des_id,
                                                                edge_feature=self.effect_list,
                                                                edge_mask=self.mask_validate, label=self.label,
                                                                combined_edge=self.combined_edge,
                                                                train_mask=self.mask_train,
                                                                valide_mask=self.mask_validate,
                                                                test_mask=self.mask_test, cost_usd=self.cost_usd_list)

        self.data_for_test = self.form_data.formulation(task_id=self.task_embedding_list,
                                                        query_feature=self.query_embedding_list,
                                                        llm_feature=self.llm_description_embedding,
                                                        org_node=self.edge_org_id,
                                                        des_node=self.edge_des_id,
                                                        edge_feature=self.effect_list, edge_mask=self.mask_test,
                                                        label=self.label, combined_edge=self.combined_edge,
                                                        train_mask=self.mask_train, valide_mask=self.mask_validate,
                                                        test_mask=self.mask_test, cost_usd=self.cost_usd_list)
        self.GNN_predict.train_validate(data=self.data_for_GNN_train, data_validate=self.data_for_GNN_validate,data_for_test=self.data_for_test, query_task_ids=self.query_task_ids)

    def test_GNN(self):
        predicted_result = self.GNN_predict.test(data=self.data_for_test,model_path=self.config['model_path'])




if __name__ == "__main__":
    import wandb
    with open("configs/config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    wandb_key = config['wandb_key']
    wandb.login(key=wandb_key)
    wandb.init(project="graph_router")
    graph_router_prediction(router_data_path=config['saved_router_data_path'],llm_path=config['llm_description_path'],
                            llm_embedding_path=config['llm_embedding_path'],config=config,wandb=wandb)
