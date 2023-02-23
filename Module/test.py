import os

import numpy as np
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from config import Config
from data import PPI_Data
from model import PPIGraph
import argparse
import pickle
import torch

config = Config()


def train_one_epoch(model, data_loader, optimizer):
    model.train()
    epoch_loss_train = 0.0
    n = 0
    for data in data_loader:
        DistanceMap, PSSM, HMM, Probert, Label = data
        Label = Label.T

        if torch.cuda.is_available():
            DistanceMap = Variable(DistanceMap.cuda().float(), requires_grad=True)
            PSSM = Variable(PSSM.cuda().float(), requires_grad=True)
            HMM = Variable(HMM.cuda().float(), requires_grad=True)
            Probert = Variable(Probert.cuda().float(), requires_grad=True)
            Label = Variable(Label.cuda(), requires_grad=False)
        else:
            DistanceMap = Variable(DistanceMap.float(), requires_grad=True)
            PSSM = Variable(PSSM.float(), requires_grad=True)
            HMM = Variable(HMM.float(), requires_grad=True)
            Probert = Variable(Probert.float(), requires_grad=True)
            Label = Variable(Label, requires_grad=False)

        PSSM = torch.squeeze(PSSM)
        HMM = torch.squeeze(HMM)
        DistanceMap = torch.squeeze(DistanceMap)
        Probert = torch.squeeze(Probert)
        Label = torch.squeeze(Label)
        y_pred = model(PSSM, HMM, DistanceMap, Probert)  # y_pred.shape = (L,2)
        # calculate loss

        loss = config.loss_fun(y_pred, Label)
        optimizer.zero_grad()
        epoch_loss_train += loss.item()

        # backward gradient
        loss.backward()

        # update all parameters
        optimizer.step()

        n += 1

    epoch_loss_train_avg = epoch_loss_train / n
    return epoch_loss_train_avg


def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}

    for data in data_loader:
        with torch.no_grad():
            DistanceMap, PSSM, HMM, Probert, Label = data

            if torch.cuda.is_available():
                DistanceMap = Variable(DistanceMap.cuda().float())
                PSSM = Variable(PSSM.cuda().float())
                HMM = Variable(HMM.cuda().float())
                Probert = Variable(Probert.cuda().float())
                Label = Variable(Label.cuda())
            else:
                DistanceMap = Variable(DistanceMap.float())
                PSSM = Variable(PSSM.float())
                HMM = Variable(HMM.float())
                Probert = Variable(Probert.float())
                Label = Variable(Label)
            PSSM = torch.squeeze(PSSM)
            HMM = torch.squeeze(HMM)
            DistanceMap = torch.squeeze(DistanceMap)
            Probert = torch.squeeze(Probert)
            Label = torch.squeeze(Label)

            y_pred = model(PSSM, HMM, DistanceMap, Probert)  # y_pred.shape = (L,2)
            loss = config.loss_fun(y_pred, Label)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            Label = Label.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(Label)
            # pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]

            epoch_loss += loss.item()
            n += 1
    epoch_loss_avg = epoch_loss / n
    model.train()

    return epoch_loss_avg, valid_true, valid_pred


def analysis(y_true, y_pred, best_threshold=None):
    if best_threshold == None:
        best_F1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            # precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, binary_pred)
            # AUPRC = metrics.auc(recalls, precisions)
            if f1 > best_F1:
                best_F1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results


def test(model, test_loader):
    _, test_true, test_pred = evaluate(model, test_loader)
    result_test = analysis(test_true, test_pred)
    print(
        "Test binary acc={},Test precision={},Test recall={},Test f1={},Test AUC={},Test AUPRC={},Test mcc={},Test_Threshold={}".format(
            result_test['binary_acc'], result_test['precision'], result_test['recall'],
            result_test['f1'], result_test['AUC'], result_test['AUPRC'], result_test['mcc'], result_test['threshold']))
    return result_test['binary_acc'], result_test['precision'], result_test['recall'], result_test['f1'], result_test[
        'AUC'], result_test['AUPRC'], result_test['mcc'], result_test['threshold']


def train(model, train_dataframe, valid_dataframe, args, fold=0):
    if args.dataset == '737':
        train_loader = DataLoader(dataset=PPI_Data(config.data_732, train_dataframe['ID'].values),
                                  batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
        valid_loader = DataLoader(dataset=PPI_Data(config.data_732, valid_dataframe['ID'].values),
                                  batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)

    best_epoch = 0
    best_val_auc = 0
    best_val_aupr = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, nesterov=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_lr)
    for epoch in range(config.epochs):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        epoch_loss_train_avg = train_one_epoch(model, train_loader, optimizer)
        # print(list(model.block1[0].named_parameters())[1])
        print("========== Evaluate Train set ==========")
        _, train_true, train_pred = evaluate(model, train_loader)
        result_train = analysis(train_true, train_pred)
        print(
            "Train loss:={},Train binary acc={},Train precision={},Train recall={},Train f1={},Train AUC={},Train AUPRC={},Train mcc={}，Train_Threshold={}".format(
                epoch_loss_train_avg, result_train['binary_acc'], result_train['precision'], result_train['recall'],
                result_train['f1'], result_train['AUC'], result_train['AUPRC'], result_train['mcc'],
                result_train['threshold']))

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred = evaluate(model, valid_loader)
        result_valid = analysis(valid_true, valid_pred)
        print(
            "Valid loss:={},Valid binary acc={},Valid precision={},Valid recall={},Valid f1={},Valid AUC={},Valid AUPRC={},Valid mcc={},Valid Threshold={}".format(
                epoch_loss_valid_avg, result_valid['binary_acc'], result_valid['precision'], result_valid['recall'],
                result_valid['f1'], result_valid['AUC'], result_valid['AUPRC'], result_valid['mcc'],
                result_valid['threshold']))
        scheduler.step()

        if best_val_aupr < result_valid['AUPRC']:
            best_epoch = epoch + 1
            best_val_auc = result_valid['AUC']
            best_val_aupr = result_valid['AUPRC']
            torch.save(model.state_dict(),
                       os.path.join(config.save_path, args.dataset + 'Fold' + str(fold) + '_best_model.pkl'))
    return best_epoch, best_val_auc, best_val_aupr


def cross_validation(all_dataframe, args, fold_number=5):
        # print("Random seed:", SEED)
        # print("Embedding type:", EMBEDDING)
        # print("Map type:", MAP_TYPE)
        # print("Map cutoff:", MAP_CUTOFF)
        # print("Feature dim:", INPUT_DIM)
        # print("Hidden dim:", HIDDEN_DIM)
        # print("Layer:", LAYER)
        # print("Dropout:", DROPOUT)
        # print("Alpha:", ALPHA)
        # print("Lambda:", LAMBDA)
        # print("Variant:", VARIANT)
        # print("Learning rate:", LEARNING_RATE)
        # print("Training epochs:", NUMBER_EPOCHS)
        # print()

    sequence_names = all_dataframe['sequence'].values
    sequence_labels = all_dataframe['label'].values
    kfold = KFold(n_splits=fold_number, shuffle=True)
    fold = 0
    best_epochs = []
    valid_aucs = []
    valid_auprs = []

    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Train on", str(train_dataframe.shape[0]), "samples, validate on", str(valid_dataframe.shape[0]),
              "samples")

        model = PPIGraph()

        if torch.cuda.is_available():
            model.cuda()

        best_epoch, valid_auc, valid_aupr = train(model, train_dataframe, valid_dataframe, args, fold + 1)
        best_epochs.append(str(best_epoch))
        valid_aucs.append(valid_auc)
        valid_auprs.append(valid_aupr)
        fold += 1

    print("\n\nBest epoch: " + " ".join(best_epochs))
    print("Average AUC of {} fold: {:.4f}".format(fold_number, sum(valid_aucs) / fold_number))
    print("Average AUPR of {} fold: {:.4f}".format(fold_number, sum(valid_auprs) / fold_number))
    Test_ACC, Test_Precision, Test_Recall, Test_F1, Test_AUC, Test_AUPRC, Test_MCC, Test_Threshold = [], [], [], [], [], [], [], []
    for f in range(fold):
        model = PPIGraph().to(config.device)
        model.load_state_dict(
            torch.load(os.path.join(config.save_path, args.dataset + 'Fold' + str(int(1 + f)) + '_best_model.pkl'),
                       map_location=config.device))
        if args.dataset == '737':
            test_loader = DataLoader(dataset=PPI_Data(config.data_732, [i.strip() for i in
                                                                        open('../dataset/737_72_164_186_315/TestId.txt',
                                                                             'r').readlines()]),
                                     batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
        ACC, Precision, Recall, F1, AUC, AUPRC, MCC, Threshold = test(model, test_loader)
        Test_ACC.append(ACC)
        Test_Precision.append(Precision)
        Test_Recall.append(Recall)
        Test_F1.append(F1)
        Test_AUC.append(AUC)
        Test_AUPRC.append(AUPRC)
        Test_MCC.append(MCC)
        Test_Threshold.append(Threshold)
    print(
        "Final! Test! Test_ACC={}±{},Test_Precision={}±{},Test_Recall={}±{},Test_F1={}±{},Test_AUC={}±{},Test_AUPRC={}±{}, Test_MCC={}±{},Test_Threshold={}".format(
            sum(Test_ACC) / len(Test_ACC), np.std(Test_ACC), sum(Test_Precision) / len(Test_Precision),
            np.std(Test_Precision),
            sum(Test_Recall) / len(Test_Recall), np.std(Test_Recall), sum(Test_F1) / len(Test_F1), np.std(Test_F1),
            sum(Test_AUC) / len(Test_AUC), np.std(Test_AUC),
            sum(Test_AUPRC) / len(Test_AUPRC), np.std(Test_AUPRC), sum(Test_MCC) / len(Test_MCC), np.std(Test_MCC),
            sum(Test_Threshold) / len(Test_Threshold)))


# def train_full_model(all_dataframe, aver_epoch):
#     print("\n\nTraining a full model using all training data...\n")
#     model = GraphPPIS(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA, VARIANT)
#     if torch.cuda.is_available():
#         model.cuda()
#
#     train_loader = DataLoader(dataset=ProDataset(all_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
#
#     for epoch in range(config.epochs):
#         print("\n========== Train epoch " + str(epoch + 1) + " ==========")
#         model.train()
#
#         epoch_loss_train_avg = train_one_epoch(model, train_loader)
#         print("========== Evaluate Train set ==========")
#         _, train_true, train_pred, _ = evaluate(model, train_loader)
#         result_train = analysis(train_true, train_pred, 0.5)
#         print("Train loss: ", epoch_loss_train_avg)
#         print("Train binary acc: ", result_train['binary_acc'])
#         print("Train AUC: ", result_train['AUC'])
#         print("Train AUPRC: ", result_train['AUPRC'])
#
#         if epoch + 1 in [aver_epoch, 45]:
#             torch.save(model.state_dict(), os.path.join(Model_Path, 'Full_model_{}.pkl'.format(epoch + 1)))


def seed_everything(seed: int):
    " refer to https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964"
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    seed_everything(22)
    parser = argparse.ArgumentParser(description='Protein-Protein interaction site prediction')
    parser.add_argument('--dataset', type=str, default='737',
                        choices=['737', '1082'])
    global args

    args = parser.parse_args()
    fold = 5
    Test_ACC, Test_Precision, Test_Recall, Test_F1, Test_AUC, Test_AUPRC, Test_MCC, Test_Threshold = [], [], [], [], [], [], [], []
    for f in range(fold):
        model = PPIGraph().to(config.device)
        model.load_state_dict(
            torch.load(os.path.join(config.save_path, args.dataset + 'Fold' + str(int(1 + f)) + '_best_model.pkl'),
                       map_location=config.device))
        if args.dataset == '737':
            test_loader = DataLoader(dataset=PPI_Data(config.data_732, [i.strip() for i in
                                                                        open('../dataset/737_72_164_186_315/TestId.txt',
                                                                             'r').readlines()]),
                                     batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
        elif args.dataset == '1082':
            test_loader = DataLoader(dataset=PPI_Data(config.data_1082, [i.strip() for i in
                                                                        open('../dataset/1082_448_634/TestId.txt',
                                                                             'r').readlines()]),
                                     batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
        ACC, Precision, Recall, F1, AUC, AUPRC, MCC, Threshold = test(model, test_loader)
        Test_ACC.append(ACC)
        Test_Precision.append(Precision)
        Test_Recall.append(Recall)
        Test_F1.append(F1)
        Test_AUC.append(AUC)
        Test_AUPRC.append(AUPRC)
        Test_MCC.append(MCC)
        Test_Threshold.append(Threshold)
    print(
        "Final! Test! Test_ACC={}±{},Test_Precision={}±{},Test_Recall={}±{},Test_F1={}±{},Test_AUC={}±{},Test_AUPRC={}±{}, Test_MCC={}±{},Test_Threshold={}".format(
            sum(Test_ACC) / len(Test_ACC), np.std(Test_ACC), sum(Test_Precision) / len(Test_Precision),
            np.std(Test_Precision),
            sum(Test_Recall) / len(Test_Recall), np.std(Test_Recall), sum(Test_F1) / len(Test_F1), np.std(Test_F1),
            sum(Test_AUC) / len(Test_AUC), np.std(Test_AUC),
            sum(Test_AUPRC) / len(Test_AUPRC), np.std(Test_AUPRC), sum(Test_MCC) / len(Test_MCC), np.std(Test_MCC),
            sum(Test_Threshold) / len(Test_Threshold)))
    Social_ACC, Social_Precision, Social_Recall, Social_F1, Social_AUC, Social_AUPRC, Social_MCC, Social_Threshold = [], [], [], [], [], [], [], []
    for f in range(fold):
        model = PPIGraph().to(config.device)
        model.load_state_dict(
            torch.load(os.path.join(config.save_path, args.dataset + 'Fold' + str(int(1 + f)) + '_best_model.pkl'),
                       map_location=config.device))
        test_loader = DataLoader(dataset=PPI_Data(config.SocialData, [i.strip() for i in
                                                                      open(config.SocialDataID).readlines()]),
                                 batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
        ACC, Precision, Recall, F1, AUC, AUPRC, MCC, Threshold = test(model, test_loader)
        Social_ACC.append(ACC)
        Social_Precision.append(Precision)
        Social_Recall.append(Recall)
        Social_F1.append(F1)
        Social_AUC.append(AUC)
        Social_AUPRC.append(AUPRC)
        Social_MCC.append(MCC)
        Social_Threshold.append(Threshold)
    print(
        "Final!  ON! SOCIAL Test_ACC={}±{},Test_Precision={}±{},Test_Recall={}±{},Test_F1={}±{},Test_AUC={}±{},Test_AUPRC={}±{}, Test_MCC={}±{},Test_Threshold={}".format(
            sum(Social_ACC) / len(Social_ACC), np.std(Social_ACC), sum(Social_Precision) / len(Social_Precision),
            np.std(Social_Precision),
            sum(Social_Recall) / len(Social_Recall), np.std(Social_Recall), sum(Social_F1) / len(Social_F1),
            np.std(Social_F1),
            sum(Social_AUC) / len(Social_AUC), np.std(Social_AUC),
            sum(Social_AUPRC) / len(Social_AUPRC), np.std(Social_AUPRC), sum(Social_MCC) / len(Social_MCC),
            np.std(Social_MCC),
            sum(Test_Threshold) / len(Test_Threshold)))
    Social_ACC, Social_Precision, Social_Recall, Social_F1, Social_AUC, Social_AUPRC, Social_MCC, Social_Threshold = [], [], [], [], [], [], [], []
    for f in range(fold):
        model = PPIGraph().to(config.device)
        model.load_state_dict(
            torch.load(os.path.join(config.save_path, args.dataset + 'Fold' + str(int(1 + f)) + '_best_model.pkl'),
                       map_location=config.device))
        test_loader = DataLoader(dataset=PPI_Data(config.Independent, [i.strip() for i in
                                                                      open(config.Independent_test).readlines()]),
                                 batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
        ACC, Precision, Recall, F1, AUC, AUPRC, MCC, Threshold = test(model, test_loader)
        Social_ACC.append(ACC)
        Social_Precision.append(Precision)
        Social_Recall.append(Recall)
        Social_F1.append(F1)
        Social_AUC.append(AUC)
        Social_AUPRC.append(AUPRC)
        Social_MCC.append(MCC)
        Social_Threshold.append(Threshold)
    print(
        "Final!  ON! INDEPENDENT Test_ACC={}±{},Test_Precision={}±{},Test_Recall={}±{},Test_F1={}±{},Test_AUC={}±{},Test_AUPRC={}±{}, Test_MCC={}±{},Test_Threshold={}".format(
            sum(Social_ACC) / len(Social_ACC), np.std(Social_ACC), sum(Social_Precision) / len(Social_Precision),
            np.std(Social_Precision),
            sum(Social_Recall) / len(Social_Recall), np.std(Social_Recall), sum(Social_F1) / len(Social_F1),
            np.std(Social_F1),
            sum(Social_AUC) / len(Social_AUC), np.std(Social_AUC),
            sum(Social_AUPRC) / len(Social_AUPRC), np.std(Social_AUPRC), sum(Social_MCC) / len(Social_MCC),
            np.std(Social_MCC),
            sum(Test_Threshold) / len(Test_Threshold)))



if __name__ == "__main__":
    main()
