import torch
from torchtext.datasets import AG_NEWS
from transformers import BertJapaneseTokenizer, BertForMaskedLM, AutoTokenizer
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader



def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


def main_sample():
    # train_iter = AG_NEWS(split='train')
    BATCH_SIZE = 64 # batch size for training
    train_iter, test_iter = AG_NEWS()
    train_dataset = list(train_iter)
    test_dataset = list(test_iter)
    # num_train = int(len(train_dataset))
    # split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=collate_batch)
    # valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
    #                             shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=collate_batch)

    print(train_dataset[0])
    print(len(train_dataset))
    # print(train_iter)
    # print(type[train_iter])
    # print(next(train_iter))


    tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

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

    # print(random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))[0])

    from transformers import BertForSequenceClassification, Trainer, TrainingArguments

    model = BertForSequenceClassification.from_pretrained("bert-large-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased", use_fast=True)

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
        train_dataset=train_dataloader,         # training dataset
        eval_dataset=test_dataloader,            # evaluation dataset
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.evaluate()




if __name__ == '__main__':
    main_sample()
