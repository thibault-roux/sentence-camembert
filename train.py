from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator



def get_data():
    dataset_id = "stsb_multi_mt"
    dataset = load_dataset(dataset_id, name="fr") #, split="test")

    train_examples = []
    train_data = dataset['train']
    for i in range(len(dataset['train'])):
        example = train_data[i]
        train_examples.append(InputExample(texts=[example['sentence1'], example['sentence2']], label=example['similarity_score']))
    # do the same thing automatically for train, dev, test
    dev_examples = []
    dev_data = dataset['dev']
    for i in range(len(dataset['dev'])):
        example = dev_data[i]
        dev_examples.append(InputExample(texts=[example['sentence1'], example['sentence2']], label=example['similarity_score']))
    test_examples = []
    test_data = dataset['test']
    for i in range(len(dataset['test'])):
        example = test_data[i]
        test_examples.append(InputExample(texts=[example['sentence1'], example['sentence2']], label=example['similarity_score']))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
    dev_dataloader = DataLoader(dev_examples, shuffle=False, batch_size=64)
    test_dataloader = DataLoader(test_examples, shuffle=False, batch_size=64)
    return train_dataloader, dev_dataloader, test_dataloader


# model_id = "sentence-transformers/all-MiniLM-L6-v2"
model_id = 'almanach/camembertv2-base'
model = SentenceTransformer(model_id)

train_loss = losses.CosineSimilarityLoss(model=model)

train_dataloader, dev_dataloader, test_dataloader = get_data()
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1)

# Save the model
model.save("models/camembert-stsb")

# Evaluate the model
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
model.evaluate(evaluator)
# print performance on test set
evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
model.evaluate(evaluator)
# print results
results = model.predict(test_dataloader)
print(results)

