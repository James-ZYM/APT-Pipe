import argparse, json, jsonlines, hashlib, tqdm, time, copy
import pandas as pd
import numpy as np
from openai import OpenAI
from googleapiclient import discovery
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.model_selection import train_test_split, StratifiedKFold
from googleapiclient import discovery
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from transformers import pipeline
from xgboost import XGBClassifier

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

def nlp_injection(metric, row):
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
            'Words': row['topic_representation']
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

    df_raw['topic_embedding'] = [topic['embedding'] for topic in text_topics]
    df_raw['topic_representation'] = [topic['representation'] for topic in text_topics]

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

    termination = False

    while len(marked_metrics) < len(STEP_3_NLP_METRICS) and not termination:

        # =========================================================================
        # Step 3.1: Split train set and validation set
        df_raw['tuned_prompt'] = tuned_prompts
        df_raw['gpt_label'] = tuned_gpt_labels
        df_raw['gpt_corect'] = df_raw.apply(lambda row: 1 if row['label'].lower()==row['gpt_label'].lower() else 0, axis=1)
        for label in df_raw['label'].unique():
            df_raw['one-hot_'+label] = df_raw['gpt_label'].apply(lambda x: 1 if x.lower()==label.lower() else 0)
        one_hot_cols = [col for col in df_raw.columns if 'one-hot' in col]
        df_train, df_validation = train_test_split(df_raw, stratify=df_raw['label'], test_size=STEP_3_VALIDATION_SET_PROPORTION, random_state=RANDOM_SEED)
        
        # =========================================================================
        # Step 3.2: Rank NLP Metrics

        fscore_xgb = {}
        for metric in STEP_3_NLP_METRICS:
            if metric not in marked_metrics:
                fscore_xgb[metric] = []
        X=df_train.drop(columns=['gpt_corect'])
        y=df_train['gpt_corect']
        skf = StratifiedKFold(n_splits=10, random_state=RANDOM_SEED, shuffle=True)
        # skf.get_n_splits(X=X, y=y)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            train = df_train.iloc[train_index]
            test = df_train.iloc[test_index]

            train_set = {}
            train_set['y'] = train['gpt_corect']
            for metric in fscore_xgb.keys():
                if metric == 'sentiment':
                    train_set[metric] = train[one_hot_cols+['sentiment_negative', 'sentiment_neutral', 'sentiment_positive']]
                if metric == 'emotion':
                    train_set[metric] = train[one_hot_cols+['emotion_anger', 'emotion_disgust', 'emotion_fear', 'emotion_joy', 'emotion_neutral', 'emotion_sadness', 'emotion_surprise']]
                if metric == 'toxicity':
                    train_set[metric] = train[one_hot_cols+['toxicity_overall', 'toxicity_severe', 'toxicity_identity', 'toxicity_insult', 'toxicity_profanity', 'toxicity_threat']]
                if metric == 'topic':
                    one_hot_codes = train[one_hot_cols].to_numpy().tolist()
                    topic_embeddings = train['topic_embedding'].to_numpy().tolist()
                    train_set['topic'] = [one_hot_codes[i]+topic_embeddings[i] for i in range(len(one_hot_codes))]

            test_set = {}
            test_set['y'] = test['gpt_corect']
            for metric in fscore_xgb.keys():
                if metric == 'sentiment':
                    test_set[metric] = test[one_hot_cols+['sentiment_negative', 'sentiment_neutral', 'sentiment_positive']]
                if metric == 'emotion':
                    test_set[metric] = test[one_hot_cols+['emotion_anger', 'emotion_disgust', 'emotion_fear', 'emotion_joy', 'emotion_neutral', 'emotion_sadness', 'emotion_surprise']]
                if metric == 'toxicity':
                    test_set[metric] = test[one_hot_cols+['toxicity_overall', 'toxicity_severe', 'toxicity_identity', 'toxicity_insult', 'toxicity_profanity', 'toxicity_threat']]
                if metric == 'topic':
                    one_hot_codes = test[one_hot_cols].to_numpy().tolist()
                    topic_embeddings = test['topic_embedding'].to_numpy().tolist()
                    test_set['topic'] = [one_hot_codes[i]+topic_embeddings[i] for i in range(len(one_hot_codes))]

            sample_weights = compute_sample_weight(
                class_weight='balanced',
                y=train_set['y']
            )

            for metric in fscore_xgb.keys():
                xgb = XGBClassifier(objective='binary:logistic', seed=RANDOM_SEED)
                xgb.fit(X=train_set[metric], y=train_set['y'], sample_weight=sample_weights)
                fscore_xgb[metric].append(f1_score(xgb.predict(test_set[metric]), test_set['y'], average='weighted'))

        fscore_mean = {}
        for metric in fscore_xgb:
            fscore_mean[metric] = np.mean(fscore_xgb[metric])
        fscore_mean = dict(sorted(fscore_mean.items(), key=lambda item: item[1], reverse=True))
        ranked_metrics = list(fscore_mean.keys())

        # =========================================================================
        # Step 3.3: Validate NLP injections

        termination = True

        for metric in ranked_metrics:
            injected_prompts = []
            for _, row in df_validation.iterrows():
                prompt = copy.deepcopy(row['tuned_prompt'])
                if len(marked_metrics) == 0:
                    tmp = json.loads(prompt[1]['content'])
                    tmp['NLP metrics'] = {'Introduction': 'Refer to the following NLP metrics of the text to make classification.'}
                tmp['NLP metrics'][metric.capitalize()] = nlp_injection(metric, row)
                prompt[1]['content'] = json.dumps(tmp, indent=4)
                injected_prompts.append(prompt)

            injection_labels = []
            for prompt in tqdm.tqdm(injected_prompts):
                response = openai_client.chat.completions.create(
                                model=GPT_VERSION,
                                response_format={ "type": "json_object" },
                                messages=prompt,
                                temperature=0
                            )
                try:
                    injection_labels.append(json.loads(response.choices[0].message.content)['Label'])
                except:
                    injection_labels.append(np.nan)

            if f1_score(df_validation['label'], injection_labels, average='weighted') > f1_score(df_validation['label'], df_validation['gpt_label'], average='weighted'):
                
                termination = False

                # =========================================================================
                # Step 3.4 and Iteration: Mark metric and go to next iteration

                for prompt in tuned_prompts:
                    if len(marked_metrics) == 0:
                        tmp = json.loads(prompt[1]['content'])
                        tmp['NLP metrics'] = {'Introduction': 'Refer to the following NLP metrics of the text to make classification.'}
                    tmp['NLP metrics'][metric.capitalize()] = nlp_injection(metric, row)
                    prompt[1]['content'] = json.dumps(tmp, indent=4)

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

                marked_metrics.append(metric)
                
                if CACHE_FLAG:
                    jsonlines.open('./data_cache/'+hashlib.sha256(file_path.encode()).hexdigest()+' (step_3_tuned).jsonl', 'w').write_all([{'marked_metrics': marked_metrics, 'tuned_prompts': tuned_prompts, 'tuned_gpt_labels': tuned_gpt_labels}])
                break
    
    return tuned_prompts
                

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
    print('step_1 finished')

    step_2_prompts = step_2(file_path, step_1_prompts)
    print('step_2 finished')

    tuned_prompts = step_3(file_path, step_2_prompts)
    print('step_3 finished')

    jsonlines.open('./output/'+hashlib.sha256(file_path.encode()).hexdigest()+' (tuned_prompts).jsonl', 'w').write_all(tuned_prompts)
    return

if __name__ == "__main__":
    main()
