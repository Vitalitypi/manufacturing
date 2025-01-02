from model.MSTAGNN import Network
from argument import get_arguments
import torch
import torch.nn as nn
from utils.util import init_seed
from utils.dataloader import get_tod_dow
from utils.norm import StandardScaler
from sklearn.preprocessing import StandardScaler as SdS
from model.Diagnosis import MLP
import numpy as np
class Inference(object):
    def __init__(self,dataset="PEMS08"):
        super(Inference, self).__init__()
        self.dataset = dataset
        self.args = get_arguments(dataset=self.dataset)
        self.model = self.get_model()
        self.data,self.scaler = self.get_dataset()
        self.dmodel,self.dx,self.dy = self.get_daignosis()

    def init_model(self, model):
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
        return model

    def get_model(self):
        if self.args.random:
            self.args.seed = torch.randint(10000, (1,))
        init_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.set_device(int(self.args.device[5]))
        else:
            self.args.device = 'cpu'
        model = Network(self.args)
        model = model.to(self.args.device)
        model = self.init_model(model)
        model.load_state_dict(torch.load('./trained/{}/best_model.pth'.format(self.args.dataset)))
        return model

    def get_daignosis(self):
        # 加载故障数据集
        data = np.load("./dataset/Diagnosis/data.npz")['data']
        # 分离特征和标签
        X = data[:, 1:].astype(float)
        y = data[:, 0].astype(int)
        # 数据标准化
        scaler = SdS()
        X_scaled = torch.from_numpy(scaler.fit_transform(X)).float().to(self.args.device)
        # 将标签转换为one-hot编码
        y_onehot = np.zeros((len(y), 7))
        y_onehot[np.arange(len(y)), y] = 1
        # 加载诊断模型
        model = MLP()
        model = model.to(self.args.device)
        model.load_state_dict(torch.load('./trained/Diagnosis/best_model.pth'))
        return model,X_scaled,y_onehot

    def get_dataset(self, periods=288):
        # load data
        data = np.load('./dataset/{}/{}.npz'.format(self.dataset, self.dataset))['data'][...,:1]
        print(data.shape)
        time_stamps, num_nodes, _ = data.shape
        time_features = get_tod_dow(self.dataset,time_stamps,num_nodes,periods)
        data = np.concatenate([data,time_features[...,:2]],axis=-1)
        mean = data[..., 0].mean()
        std = data[..., 0].std()
        scaler = StandardScaler(mean, std)
        data[..., 0] = scaler.transform(data[..., 0])
        data = torch.from_numpy(data).unsqueeze(0).float().to(self.args.device)
        return data,scaler

    def inference(self, idx = 0,horizon = 12):
        if idx+12>=self.data.shape[1]:
            idx = 0
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.data[:,idx:idx+horizon])
            true = self.data[:,idx+horizon:idx+2*horizon]
            pred = pred[...,0].squeeze(0)
            true = true[...,0].squeeze(0)
            y_pred = self.scaler.inverse_transform(pred).t()
            y_true = self.scaler.inverse_transform(true).t()
            pred_list = y_pred.cpu().numpy().tolist()
            true_list = y_true.cpu().numpy().tolist()
            return pred_list,true_list

    def diagnosis(self, idx = 0):
        self.dmodel.eval()
        with torch.no_grad():
            pred = self.dmodel(self.dx[:20]).cpu().numpy()
            pred = np.argmax(pred, axis=1).tolist()
            true = np.argmax(self.dy[:20], axis=1).tolist()
            print(pred,true)
            return pred,true