from model.MSTAGNN import Network
from argument import get_arguments
import torch
import torch.nn as nn
from utils.util import init_seed
from utils.dataloader import get_tod_dow
from utils.norm import StandardScaler
import numpy as np
class Inference(object):
    def __init__(self,dataset="PEMS08"):
        super(Inference, self).__init__()
        self.dataset = dataset
        self.args = get_arguments(dataset=self.dataset)
        self.model = self.get_model()
        self.data,self.scaler = self.get_dataset()

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
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.data[:,idx:idx+horizon])
            y_pred = self.scaler.inverse_transform(pred)
            y_true = self.scaler.inverse_transform(self.data[:,idx+horizon:idx+2*horizon])
            return y_pred,y_true
