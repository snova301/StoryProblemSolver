# import torch
# from torchtext.datasets import AG_NEWS
# from transformers import BertJapaneseTokenizer, BertForMaskedLM, AutoTokenizer
# from torch.utils.data.dataset import random_split
# from torch.utils.data import DataLoader

import os
from glob import glob
import pandas as pd
import linecache



import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext

from transformers import BertModel
# from transformers import BertTokenizer
from transformers import AutoTokenizer





def main_sample():
    # 事前に
    # wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
    # tar xvf ldcc-20140209.tar.gz

    '''
    # カテゴリを配列で取得
    categories = [name for name in os.listdir("text") if os.path.isdir("text/" + name)]
    print(categories)
    # ['movie-enter', 'it-life-hack', 'kaden-channel', 'topic-news', 'livedoor-homme', 'peachy', 'sports-watch', 'dokujo-tsushin', 'smax']

    datasets = pd.DataFrame(columns=["title", "category"])
    for cat in categories:
        path = "text/" + cat + "/*.txt"
        files = glob(path)
        for text_name in files:
            title = linecache.getline(text_name, 3)
            s = pd.Series([title, cat], index=datasets.columns)
            datasets = datasets.append(s, ignore_index=True)

    # データフレームシャッフル
    datasets = datasets.sample(frac=1).reset_index(drop=True)
    datasets.to_pickle('./text/livedoor_title_category.pickle')


    # データセット格納先
    with open("./text/livedoor_title_category.pickle", 'rb') as f:
        livedoor_data = pickle.load(f)

    print(livedoor_data.head())
    '''

    # データ読み込み
    with open('./text/livedoor_title_category.pickle', 'rb') as r:
        livedoor_data = pickle.load(r)

    # データ確認
    # print(livedoor_data.head())

    # 正解ラベル（カテゴリー）をデータセットから取得
    categories = list(set(livedoor_data['category']))
    # print(categories)

    # カテゴリーのID辞書を作成
    id2cat = dict(zip(list(range(len(categories))), categories))
    cat2id = dict(zip(categories, list(range(len(categories)))))
    # print(id2cat)
    # print(cat2id)

    # DataFrameにカテゴリーID列を追加
    livedoor_data['category_id'] = livedoor_data['category'].map(cat2id)

    # 念の為シャッフル
    livedoor_data = livedoor_data.sample(frac=1).reset_index(drop=True)

    # データセットを本文とカテゴリーID列だけにする
    livedoor_data = livedoor_data[['title', 'category_id']]
    print(livedoor_data)
    # display(livedoor_data.head())

    train_df, test_df = train_test_split(livedoor_data, train_size=0.8)
    print("学習データサイズ", train_df.shape[0])
    print("テストデータサイズ", test_df.shape[0])
    #学習データサイズ 5900
    #テストデータサイズ 1476

    # tsvファイルとして保存する
    train_df.to_csv('./train.tsv', sep='\t', index=False, header=None)
    test_df.to_csv('./test.tsv', sep='\t', index=False, header=None)


    tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

    # BERTはサブワードを含めて最大512単語まで扱える
    MAX_LENGTH = 512
    def bert_tokenizer(text):
        return tokenizer.encode(text, max_length=MAX_LENGTH, truncation=True, return_tensors='pt')[0]

    TEXT = torchtext.legacy.data.Field(sequential=True, tokenize=bert_tokenizer, use_vocab=False, lower=False,
                                include_lengths=True, batch_first=True, fix_length=MAX_LENGTH, pad_token=0)
    LABEL = torchtext.legacy.data.Field(sequential=False, use_vocab=False)

    train_data, test_data = torchtext.legacy.data.TabularDataset.splits(
        path='./', train='train.tsv', test='test.tsv', format='tsv', fields=[('Text', TEXT), ('Label', LABEL)])

    # BERTではミニバッチサイズは16か32が推奨される
    BATCH_SIZE = 16
    train_iter, test_iter = torchtext.legacy.data.Iterator.splits((train_data, test_data), batch_sizes=(BATCH_SIZE, BATCH_SIZE), repeat=False, sort=False)

    classifier = BertClassifier()

    # まずは全部OFF
    for param in classifier.parameters():
        param.requires_grad = False

    # BERTの最終4層分をON
    for param in classifier.bert.encoder.layer[-1].parameters():
        param.requires_grad = True

    for param in classifier.bert.encoder.layer[-2].parameters():
        param.requires_grad = True

    for param in classifier.bert.encoder.layer[-3].parameters():
        param.requires_grad = True

    for param in classifier.bert.encoder.layer[-4].parameters():
        param.requires_grad = True

    # クラス分類のところもON
    for param in classifier.linear.parameters():
        param.requires_grad = True

    # 事前学習済の箇所は学習率小さめ、最後の全結合層は大きめにする。
    optimizer = optim.Adam([
        {'params': classifier.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
        {'params': classifier.bert.encoder.layer[-2].parameters(), 'lr': 5e-5},
        {'params': classifier.bert.encoder.layer[-3].parameters(), 'lr': 5e-5},
        {'params': classifier.bert.encoder.layer[-4].parameters(), 'lr': 5e-5},
        {'params': classifier.linear.parameters(), 'lr': 1e-4}
    ])

    # 損失関数の設定
    loss_function = nn.NLLLoss()

    # GPUの設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ネットワークをGPUへ送る
    classifier.to(device)
    losses = []

    # エポック数は5で
    for epoch in range(5):

        all_loss = 0

        for idx, batch in enumerate(train_iter):

            classifier.zero_grad()

            input_ids = batch.Text[0].to(device)
            label_ids = batch.Label.to(device)

            out, _ = classifier(input_ids)

            batch_loss = loss_function(out, label_ids)
            batch_loss.backward()

            optimizer.step()

            all_loss += batch_loss.item()

        print("epoch", epoch, "\t" , "loss", all_loss)


    answer = []
    prediction = []

    with torch.no_grad():
        for batch in test_iter:

            text_tensor = batch.Text[0].to(device)
            label_tensor = batch.Label.to(device)

            score, _ = classifier(text_tensor)
            _, pred = torch.max(score, 1)

            prediction += list(pred.cpu().numpy())
            answer += list(label_tensor.cpu().numpy())

    print(classification_report(prediction, answer, target_names=categories))





class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()

        # 日本語学習済モデルをロードする
        # output_attentions=Trueで順伝播のときにattention weightを受け取れるようにする
        # output_hidden_state=Trueで12層のBertLayerの隠れ層を取得する
        self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking',
                                              output_attentions=True,
                                              output_hidden_states=True)

        # BERTの隠れ層の次元数は768だが、最終4層分のベクトルを結合したものを扱うので、７６８×4次元としている。
        self.linear = nn.Linear(768*4, 9)

        # 重み初期化処理
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    # clsトークンのベクトルを取得する用の関数を用意
    def _get_cls_vec(self, vec):
        return vec[:,0,:].view(-1, 768)

    def forward(self, input_ids):

        # 順伝播の出力結果は辞書形式なので、必要な値のkeyを指定して取得する
        output = self.bert(input_ids)
        attentions = output['attentions']
        hidden_states = output['hidden_states']

        # 最終４層の隠れ層からそれぞれclsトークンのベクトルを取得する
        vec1 = self._get_cls_vec(hidden_states[-1])
        vec2 = self._get_cls_vec(hidden_states[-2])
        vec3 = self._get_cls_vec(hidden_states[-3])
        vec4 = self._get_cls_vec(hidden_states[-4])

        # 4つのclsトークンを結合して１つのベクトルにする。
        vec = torch.cat([vec1, vec2, vec3, vec4], dim=1)

        # 全結合層でクラス分類用に次元を変換
        out = self.linear(vec)

        return F.log_softmax(out, dim=1), attentions






if __name__ == '__main__':
    main_sample()
