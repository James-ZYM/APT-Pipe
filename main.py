import argparse, json, jsonlines, hashlib, tqdm, time
import pandas as pd
import numpy as np
from openai import OpenAI
from googleapiclient import discovery
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.model_selection import train_test_split
from googleapiclient import discovery
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from transformers import pipeline

sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment", return_all_scores=True)
emotion_task = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

openai_client = OpenAI(api_key = 'OPENAI_API_KEY')
google_client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey='GOOGLE_API_KEY',
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
)

CACHE_FLAG = True
GPT_VERSION = 'gpt-3.5-turbo-0125'
RANDOM_SEED = 42
STEP_2_SHOT_NUMBER = 3
STEP_2_VALIDATION_SET_PROPORTION = 1.0
STEP_3_VALIDATION_SET_PROPORTION = 0.25
STEP_3_NLP_METRICS = ['sentiment', 'emotion', 'toxicity', 'topic']

def step_1(file_path, task_domain):
    if CACHE_FLAG:
        try:
            return [item for item in jsonlines.open('./data_cache/'+hashlib.sha256(file_path.encode()).hexdigest()+' (step_1_prompts).jsonl', 'r')]
        except:
            pass

    df_raw = pd.read_csv(file_path)
    labels = df_raw['label'].unique()

    step_1_prompts = []
    for ind, row in df_raw.iterrows():
        prompt = json.dumps({
                "Prompt": "Classify the following text by given labels for specified task.",
                "Text": row['text'],
                "Task": task_domain,
                "Labels": list(labels),
                "Desired format": {
                "Label": "<label_for_classification>"
                }
            }, indent=4)
        step_1_prompts.append({'true_label': row['label'], 'row_ind': ind, 'prompt': [{"role": "user", "content": prompt}]})
    
    if CACHE_FLAG:
        jsonlines.open('./data_cache/'+hashlib.sha256(file_path.encode()).hexdigest()+' (step_1_prompts).jsonl', 'w').write_all(step_1_prompts)

    return step_1_prompts

def step_2(file_path, step_1_prompts):
    df_raw = pd.read_csv(file_path)

    text_embeddings = []
    if CACHE_FLAG:
        try:
            text_embeddings = [item for item in jsonlines.open('./data_cache/'+hashlib.sha256(file_path.encode()).hexdigest()+' (step_2_embeddings).jsonl', 'r')]
        except:
            pass
    
    if len(text_embeddings) == 0:
        for text in tqdm.tqdm(df_raw['text']):
            response = openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            text_embeddings.append({'text': text, 'embedding': response.data[0].embedding})

        if CACHE_FLAG:
            jsonlines.open('./data_cache/'+hashlib.sha256(file_path.encode()).hexdigest()+' (step_2_embeddings).jsonl', 'w').write_all(text_embeddings)
    
    if CACHE_FLAG:
        try:
            df_test = pd.read_csv('./data_cache/'+hashlib.sha256(file_path.encode()).hexdigest()+' (step_2_test).csv')
        except:
            df_test = df_raw.groupby('label', group_keys=False).apply(lambda x: x.sample(n=int(STEP_2_VALIDATION_SET_PROPORTION*len(df_raw)/len(df_raw['label'].unique())), random_state=RANDOM_SEED))
    
    cos_sim = cosine_similarity([item['embedding'] for item in text_embeddings], [item['embedding'] for item in text_embeddings])

    if 'gpt_few_shot_label' not in df_test.columns:

        zero_shot_prompts = []
        few_shot_prompts = []
        for ind in df_test.reset_index()['index']:
            zero_shot_prompts.append([{"role": "system", "content": "You are a helpful assistant designed to output JSON within the desired format {'Lable': <label_for_classification>}."}]+step_1_prompts[ind]['prompt'])
            prompt = json.loads(step_1_prompts[ind]['prompt'][0]['content'])
            prompt['Examples'] = [{'Text': df_raw.iloc[exemplar_ind]['text'], 'Label': df_raw.iloc[exemplar_ind]['label']} for exemplar_ind in np.argsort(cos_sim[ind])[::-1][1:-1][:STEP_2_SHOT_NUMBER]]
            few_shot_prompts.append([{"role": "system", "content": "You are a helpful assistant designed to output JSON within the desired format {'Lable': <label_for_classification>}."}, {"role": "user", "content": json.dumps(prompt, indent=4)}])

        df_test = df_test.reset_index()

        zero_shot_labels = []
        for prompt in tqdm.tqdm(zero_shot_prompts):
            response = openai_client.chat.completions.create(
                            model=GPT_VERSION,
                            response_format={ "type": "json_object" },
                            messages=prompt,
                            temperature=0
                        )
            try:
                zero_shot_labels.append(json.loads(response.choices[0].message.content)['Label'])
            except:
                zero_shot_labels.append(np.nan)
        df_test['gpt_zero_shot_label'] = zero_shot_labels

        few_shot_labels = []
        for prompt in tqdm.tqdm(few_shot_prompts):
            response = openai_client.chat.completions.create(
                            model=GPT_VERSION,
                            response_format={ "type": "json_object" },
                            messages=prompt,
                            temperature=0
                        )
            try:
                few_shot_labels.append(json.loads(response.choices[0].message.content)['Label'])
            except:
                few_shot_labels.append(np.nan)
        df_test['gpt_few_shot_label'] = few_shot_labels

        if CACHE_FLAG:
            df_test.to_csv('./data_cache/'+hashlib.sha256(file_path.encode()).hexdigest()+' (step_2_test).csv', index=False)

    step_2_prompts = []
    if f1_score(list(df_test['label']), df_test['gpt_zero_shot_label'], average='weighted') < f1_score(list(df_test['label']), df_test['gpt_few_shot_label'], average='weighted'):
        for ind in df_raw.reset_index()['index']:
            prompt = json.loads(step_1_prompts[ind]['prompt'][0]['content'])
            prompt['Examples'] = [{'Text': df_raw.iloc[exemplar_ind]['text'], 'Label': df_raw.iloc[exemplar_ind]['label']} for exemplar_ind in np.argsort(cos_sim[ind])[::-1][1:-1][:STEP_2_SHOT_NUMBER]]
            step_2_prompts.append([{"role": "system", "content": "You are a helpful assistant designed to output JSON within the desired format {'Lable': <label_for_classification>}."}, {"role": "user", "content": json.dumps(prompt, indent=4)}])
    else:
        step_2_prompts.append([{"role": "system", "content": "You are a helpful assistant designed to output JSON within the desired format {'Lable': <label_for_classification>}."}]+step_1_prompts[ind]['prompt'])
            
    if CACHE_FLAG:
        jsonlines.open('./data_cache/'+hashlib.sha256(file_path.encode()).hexdigest()+' (step_2_prompts).jsonl', 'w').write_all(step_2_prompts)

    return step_2_prompts

def nlp_injection(metric, row, representation):
    if metric == 'sentiment':
        injection = {
            'Introduction': 'Scores of sentiment leaning of text (ranging from 0 to 1).',
                'Scores': {
                    'Positive': row['sentiment_positive'],
                    'Neutral': row['sentiment_neutral'],
                    'Negative': row['sentiment_negative']
                }
            }
        
    elif metric == 'emotion':
        injection = {
            'Introduction': 'Scores of emotion leaning of text (ranging from 0 to 1).',
                'Scores': {
                    'Anger': row['emotion_anger'],
                    'Disgust': row['emotion_disgust'],
                    'Fear': row['emotion_fear'],
                    'Joy': row['emotion_joy'],
                    'Neutral': row['emotion_neutral'],
                    'Sadness': row['emotion_sadness'],
                    'Surprise': row['emotion_surprise']
                }
            }
        
    elif metric == 'toxicity':
        injection = {
            'Introduction': 'Scores of toxcity degree of text (ranging from 0 to 1).',
                'Scores': {
                    'Overall Toxicity': row['toxicity_overall'],
                    'Severe Toxicity': row['toxicity_severe'],
                    'Identity Attack': row['toxicity_identity'],
                    'Insult': row['toxicity_insult'],
                    'Profanity': row['toxicity_profanity'],
                    'Threat': row['toxicity_threat']
                }
        }
        
    elif metric == 'topic':
        injection = {
            'Introduction': 'Representative words to describe the major topic of the text.',
            'Words': representation
        }

    return injection

def step_3(file_path, step_2_prompts):

    df_raw = pd.read_csv(file_path)
    try:
        df_raw = pd.read_csv('./data_cache/'+hashlib.sha256(file_path.encode()).hexdigest()+' (step_3_metrics).csv')
    except:
        sentiment_scores = sentiment_task(list(df_raw['text']))
        df_raw['sentiment_negative'] = [score[0]['score'] for score in sentiment_scores]
        df_raw['sentiment_neutral'] = [score[1]['score'] for score in sentiment_scores]
        df_raw['sentiment_positive'] = [score[2]['score'] for score in sentiment_scores]

        emotion_scores = emotion_task(list(df_raw['text']))
        df_raw['emotion_anger'] = [score[0]['score'] for score in emotion_scores]
        df_raw['emotion_disgust'] = [score[1]['score'] for score in emotion_scores]
        df_raw['emotion_fear'] = [score[2]['score'] for score in emotion_scores]
        df_raw['emotion_joy'] = [score[3]['score'] for score in emotion_scores]
        df_raw['emotion_neutral'] = [score[4]['score'] for score in emotion_scores]
        df_raw['emotion_sadness'] = [score[5]['score'] for score in emotion_scores]
        df_raw['emotion_surprise'] = [score[6]['score'] for score in emotion_scores]

        toxicity_scores = []
        for text in df_raw['text']:
            while True:
                try:
                    toxicity_scores.append(
                        google_client.comments().analyze(body={
                            'comment': {'text': text},
                            'requestedAttributes': {'TOXICITY': {}, 'SEVERE_TOXICITY': {}, 'IDENTITY_ATTACK': {}, 'INSULT': {}, 'PROFANITY': {}, 'THREAT': {}}
                        }).execute()
                    )
                    break
                except:
                    time.sleep(5)
        df_raw['toxicity_overall'] = [score['attributeScores']['TOXICITY']['summaryScore']['value'] for score in toxicity_scores]
        df_raw['toxicity_severe'] = [score['attributeScores']['SEVERE_TOXICITY']['summaryScore']['value'] for score in toxicity_scores]
        df_raw['toxicity_identity'] = [score['attributeScores']['IDENTITY_ATTACK']['summaryScore']['value'] for score in toxicity_scores]
        df_raw['toxicity_insult'] = [score['attributeScores']['INSULT']['summaryScore']['value'] for score in toxicity_scores]
        df_raw['toxicity_profanity'] = [score['attributeScores']['PROFANITY']['summaryScore']['value'] for score in toxicity_scores]
        df_raw['toxicity_threat'] = [score['attributeScores']['THREAT']['summaryScore']['value'] for score in toxicity_scores]

        if CACHE_FLAG:
            df_raw.to_csv('./data_cache/'+hashlib.sha256(file_path.encode()).hexdigest()+' (step_3_metrics).csv', index=False)

    try:
        text_topics = [item for item in jsonlines.open('./data_cache/'+hashlib.sha256(file_path.encode()).hexdigest()+' (step_3_topics).jsonl', 'r')]
    except:
        topic_task = BERTopic(representation_model=KeyBERTInspired(), calculate_probabilities=True)
        topic_task.fit(df_raw['text'])
        topics, probs = topic_task.transform(df_raw['text'])
        representations = topic_task.get_document_info(df_raw['text'])['Representation']
        text_topics = []
        for i in range(len(probs)):
            text_topics.append({'representation': representations[i], 'embedding': probs[i].tolist()})
        
        if CACHE_FLAG: 
            jsonlines.open('./data_cache/'+hashlib.sha256(file_path.encode()).hexdigest()+' (step_3_topics).jsonl', 'w').write_all(text_topics)

    df_train, df_validation = train_test_split(df_raw, stratify=df_raw['label'], test_size=STEP_3_VALIDATION_SET_PROPORTION, random_state=RANDOM_SEED)
    
    tuned_prompts = step_2_prompts
    marked_metrics = []

    try:
        for item in jsonlines.open('./data_cache/'+hashlib.sha256(file_path.encode()).hexdigest()+' (step_3_tuned).jsonl', 'r'):
            break
        tuned_gpt_labels = item['tuned_gpt_labels']
        marked_metrics = item['marked_metrics']
        tuned_prompts = item['tuned_prompts']
    except:
        tuned_gpt_labels = []
        for prompt in tqdm.tqdm(tuned_prompts):
            response = openai_client.chat.completions.create(
                            model=GPT_VERSION,
                            response_format={ "type": "json_object" },
                            messages=prompt,
                            temperature=0
                        )
            try:
                tuned_gpt_labels.append(json.loads(response.choices[0].message.content)['Label'])
            except:
                tuned_gpt_labels.append(np.nan)
        if CACHE_FLAG:
            jsonlines.open('./data_cache/'+hashlib.sha256(file_path.encode()).hexdigest()+' (step_3_tuned).jsonl', 'w').write_all([{'marked_metrics': marked_metrics, 'tuned_prompts': tuned_prompts, 'tuned_gpt_labels': tuned_gpt_labels}])

    while True:
        if len(marked_metrics) == 0:
            for prompt in tuned_prompts:
                tmp = json.loads(prompt[1]['content'])
                tmp['NLP metrics'] = {'Introduction': 'Refer to the following NLP metrics of the text to make classification.'}
                prompt[1]['content'] = json.dumps(tmp, indent=4)
        
        
        for metric in STEP_3_NLP_METRICS:
            if metric in marked_metrics:
                continue
            break
        break

    print(tuned_prompts[0])
    return
                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='file_path', type=str, help='Path of CSV file to generate prompts.')
    parser.add_argument('-d', dest='task_domain', type=str, help='Domain for annotation task (e.g., News classification).')
    args = parser.parse_args()
    try:
        file_path = args.file_path
        task_domain = args.task_domain
    except:
        print('Error: can not read input dataset or missing task domain.')
        exit()

    step_1_prompts = step_1(file_path, task_domain)

    step_2_prompts = step_2(file_path, step_1_prompts)

    step_3(file_path, step_2_prompts)
    return

if __name__ == "__main__":
    main()