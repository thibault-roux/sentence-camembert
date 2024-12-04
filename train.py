from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch import nn

def format_data(dataset, name):
    examples = []
    for i in range(len(dataset)):
        example = dataset[i]
        examples.append(InputExample(texts=[example['sentence1'], example['sentence2']], label=example['similarity_score'] / 5))
    return examples

def get_data():
    dataset_id = "stsb_multi_mt"
    dataset = load_dataset(dataset_id, name="fr") #, split="test")

    train_examples = format_data(dataset['train'], 'train')
    # dev_examples = format_data(dataset['dev'], 'dev')
    # test_examples = format_data(dataset['test'], 'test')
    
    dev_sentences1 = [example['sentence1'] for example in dataset['validation']]
    dev_sentences2 = [example['sentence2'] for example in dataset['validation']]
    dev_scores = [example['score'] / 5.0 for example in dataset['validation']]

    test_sentences1 = [example['sentence1'] for example in dataset['test']]
    test_sentences2 = [example['sentence2'] for example in dataset['test']]
    test_scores = [example['score'] / 5.0 for example in dataset['test']]
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
    # dev_dataloader = DataLoader(dev_examples, shuffle=False, batch_size=64)
    # test_dataloader = DataLoader(test_examples, shuffle=False, batch_size=64)
    # return train_dataloader, dev_dataloader, test_dataloader
    return train_dataloader, (dev_sentences1, dev_sentences2, dev_scores), (test_sentences1, test_sentences2, test_scores)


# model_id = "sentence-transformers/all-MiniLM-L6-v2"
model_id = 'almanach/camembertv2-base'
model = SentenceTransformer(model_id)

train_loss = losses.CosineSimilarityLoss(model=model)

# train_dataloader, dev_dataloader, test_dataloader = get_data()
train_dataloader, dev_examples, test_examples = get_data()
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1)

# Save the model
model.save("models/camembert-stsb")

# Evaluate the model
evaluator = EmbeddingSimilarityEvaluator(dev_sentences1, dev_sentences2, dev_scores)
model.evaluate(evaluator)
# print performance on test set
evaluator = EmbeddingSimilarityEvaluator(test_sentences1, test_sentences2, test_scores)
model.evaluate(evaluator)

# Compute embeddings manually and print similarity scores
test_sentences1 = [example.texts[0] for example in test_examples]
test_sentences2 = [example.texts[1] for example in test_examples]
embeddings1 = model.encode(test_sentences1, convert_to_tensor=True)
embeddings2 = model.encode(test_sentences2, convert_to_tensor=True)

cos = nn.CosineSimilarity(dim=1)
similarity_scores = cos(embeddings1, embeddings2)
print(similarity_scores)
