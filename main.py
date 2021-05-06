import torch
# from torchtext.datasets import AG_NEWS
from transformers import BertJapaneseTokenizer, BertForMaskedLM

def main_sample():
    # train_iter = AG_NEWS(split='train')
    # print(train_iter)
    # print(type[train_iter])
    # print(next(train_iter))
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

    # Tokenize input
    text = 'テレビでサッカーの試合を見る。'
    tokenized_text = tokenizer.tokenize(text)
    print(tokenized_text)

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = 2
    tokenized_text[masked_index] = '[MASK]'
    # ['テレビ', 'で', '[MASK]', 'の', '試合', 'を', '見る', '。']
    print(tokenized_text)

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # [571, 12, 4, 5, 608, 11, 2867, 8]
    print(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    # tensor([[ 571,   12,    4,    5,  608,   11, 2867,    8]])
    print(tokens_tensor)

    # # Load pre-trained model
    # model = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    # model.eval()

    # # Predict
    # with torch.no_grad():
    #     outputs = model(tokens_tensor)
    #     predictions = outputs[0][0, masked_index].topk(5) # 予測結果の上位5件を抽出

    # # Show results
    # for i, index_t in enumerate(predictions.indices):
    #     index = index_t.item()
    #     token = tokenizer.convert_ids_to_tokens([index])[0]
    #     print(i, token)

    from transformers import BertForSequenceClassification, Trainer, TrainingArguments

    model = BertForSequenceClassification.from_pretrained("bert-large-uncased")

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset            # evaluation dataset
    )



if __name__ == '__main__':
    main_sample()
