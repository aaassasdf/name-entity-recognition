# name-entity-recognition
This is a NLP project focusing on NER(Name Entity Recognition). 
The dataset used is bionlp2004 from Hugging Face. It labels 
This project uses BERT-base-uncased model and also included the fine-tuning process
This task is specified to recognise Biological terminology, which are:
1. DNA
2. protein
3. cell_type
4. RNA

The dataset labels all these terms with numerical tags. The table shows as following:
{
    "O": 0,
    "B-DNA": 1,
    "I-DNA": 2,
    "B-protein": 3,
    "I-protein": 4,
    "B-cell_type": 5,
    "I-cell_type": 6,
    "B-cell_line": 7,
    "I-cell_line": 8,
    "B-RNA": 9,
    "I-RNA": 10
}

The datasets also labels consecutive words which are in same category by giving prefix "B" and "I".
"B" means it is the leading word and "I" means it is the same category of word accoring to the previous one.
e.g. "Displacement of an E-box-binding repressor..." is labelled by 0,0,0,3,4,...



credit to: Rohan-Paul-AI
youtube: https://www.youtube.com/watch?v=dzyDHMycx_c&list=PLxqBkZuBynVQEvXfJpq3smfuKq3AiNW-N&index=19
github: https://github.com/rohan-paul/MachineLearning-DeepLearning-Code-for-my-YouTube-Channel/blob/master/NLP/YT_Fine_tuning_BERT_NER_v1.ipynb

Thank you for the resources and education. 
