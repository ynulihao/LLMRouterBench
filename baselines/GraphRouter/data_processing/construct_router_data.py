from llm_engine import LLMEngine
from utils import savejson,loadjson,savepkl,loadpkl,get_embedding
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import yaml

class data_building:
    def __init__(self,qa_path,llm_path,config):
        self.qa_data=pd.read_csv(qa_path)
        self.llm_description = loadjson(llm_path)
        self.llm_names = list(self.llm_description.keys())
        self.all_llm_description = []
        for inter in self.llm_names:
            self.all_llm_description.append(self.llm_description[inter]['feature'])
        self.MyLLMEngine = LLMEngine(llm_names=self.llm_names,llm_description=self.llm_description)
        self.config=config
        self.construct_data_with_LLM()


    def construct_data_with_LLM(self):
        df = pd.DataFrame(columns=['task_id', 'query','query_embedding', 'ground_truth', 'metric','llm',
                                   'effect','cost'])
        count=0
        for row in self.qa_data.itertuples():
            task_id_t=row.task_id
            query_t=row.query
            task_description=row.task_description
            if task_id_t=="multi_news":
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                tokens = tokenizer.tokenize(query_t)
                extracted_text = tokens[:3000]
                query_t = tokenizer.convert_tokens_to_string(extracted_text)
            query_t_embedding = get_embedding([query_t])
            task_description_embedding=get_embedding([task_description])
            ground_truth_t=row.ground_truth
            metric_t=row.metric

            for a_t in range(len(self.llm_names)):
                response_t = self.MyLLMEngine.get_llm_response(query=query_t, llm_idx=a_t)
                reward_t = self.MyLLMEngine.eval(prediction=response_t, ground_truth=ground_truth_t, metric=metric_t)
                cost_t = self.MyLLMEngine.compute_cost(llm_idx=a_t, input_text=query_t, output_size=self.config['query_response_length'])
                llm_t=self.llm_names[a_t]
                new_row = {'task_id':task_id_t,'task_description':task_description,'task_description_embedding':task_description_embedding,'query':query_t,'query_embedding':query_t_embedding, 'ground_truth':ground_truth_t, 'metric':metric_t,
                           'llm':llm_t,'effect':reward_t,'cost':cost_t}
                df = df._append(new_row, ignore_index=True)
                count+=1

        # Normalize cost according to task
        df['cost'] = df.groupby('task_id')['cost'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

        df.to_csv(self.config['saved_router_data_path'], index=False)
        llm_description_embedding = get_embedding(self.all_llm_description)
        savepkl(llm_description_embedding, self.config['llm_embedding_path'])


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
    with open("configs/config.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    os.environ["TOGETHERAI_API_KEY"] = config["api_key"]
    data_building(qa_path=config['unified_qa_data_path'],llm_path=config['llm_description_path'],config=config)