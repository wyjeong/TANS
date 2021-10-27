####################################################################################################
# TANS: Task-Adaptive Neural Network Search with Meta-Contrastive Learning
# Wonyong Jeong, Hayeon Lee, Geon Park, Eunyoung Hyung, Jinheon Baek, Sung Ju Hwang
# github: https://github.com/wyjeong/TANS, email: wyjeong@kaist.ac.kr
####################################################################################################

import os
import sys
import glob
import time
import atexit
import torch 
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from ofa.model_zoo import ofa_net
from retrieval.loss import HardNegativeContrastiveLoss
from retrieval.measure import compute_recall

from misc.utils import *
from data.loader import *
from retrieval.nets import *

class Retrieval:

    def __init__(self, args):
        self.args = args
        self.parameters = []
        self.device = get_device(args)
        atexit.register(self.atexit)

    def atexit(self):
        print('Process destroyed.')

    def train(self):
        print(f'Begin train process')
        start_time = time.time()
        self.init_loaders()        
        self.init_models()
        self.train_cross_modal_space()
        self.save_cross_modal_space()
        print(f'Process done ({time.time()-start_time:.2f})')
        sys.exit()

    def init_loaders(self):
        print('==> loading data loaders ... ')
        self.tr_dataset, self.tr_loader = get_loader(self.args, mode='train')
        self.te_dataset, self.te_loader = get_loader(self.args, mode='test')

    def init_models(self):
        print('==> loading encoders ... ')
        self.enc_m = ModelEncoder(self.args).to(self.device)
        self.enc_q = QueryEncoder(self.args).to(self.device)
        self.predictor = PerformancePredictor(self.args).to(self.device)
        self.parameters = [*self.enc_q.parameters(),*self.enc_m.parameters(),*self.predictor.parameters()]
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.args.lr)
        self.criterion = HardNegativeContrastiveLoss(nmax=self.args.n_groups, contrast=True)
        self.criterion_mse = torch.nn.MSELoss()

    def train_cross_modal_space(self):
        print('==> train the cross modal space from model-zoo ... ')

        self.scores = {
            'tr_lss': [], 'te_lss': [],
            'r_1': [], 'r_5': [], 'r_10': [], 'r_50': [],
            'mean': [], 'median': [], 'mse': []}
        
        max_recall = 0
        start_time = time.time()
        for curr_epoch in range(self.args.n_epochs):
            ep_time = time.time()
            self.curr_epoch = curr_epoch
            
            ##################################################
            self.optimizer.zero_grad()
            dataset, acc, topol, f_emb = next(iter(self.tr_loader)) # 1 iteration == 1 epoch 
            query = self.tr_dataset.get_query(dataset)
            q, m, lss, lss_mse = self.forward(acc, topol, f_emb, query)
            lss = lss + lss_mse
            lss.backward()
            self.optimizer.step()
            ##################################################
            
            tr_lss = lss.item()
            te_lss, R, medr, meanr, mse = self.evaluate()
            print(  f'ep:{self.curr_epoch}, ' +
                    f'mse:{mse:.3f} ' +
                    f'tr_lss: {tr_lss:.3f}, ' +
                    f'te_lss:{te_lss:.3f}, ' +
                    f'R@1 {R[1]:.1f} ({max_recall:.1f}), R@5 {R[5]:.1f}, R@10 {R[10]:.1f}, R@50 {R[50]:.1f} ' +
                    f'mean {meanr:.1f}, median {medr:.1f} ({time.time()-ep_time:.2f})')

            self.scores['tr_lss'].append(tr_lss)
            self.scores['te_lss'].append(te_lss)
            self.scores['r_1'].append(R[1])
            self.scores['r_5'].append(R[5])
            self.scores['r_10'].append(R[10])
            self.scores['r_50'].append(R[50])
            self.scores['median'].append(medr)
            self.scores['mean'].append(meanr)
            self.scores['mse'].append(mse)
            self.save_scroes()

            if R[1] > max_recall:
                max_recall = R[1]
                self.save_model(True, curr_epoch, R, medr, meanr, mse)
            
        self.save_model(False, curr_epoch, R, medr, meanr, mse)
        self.save_scroes()
        print(f'==> training the cross modal space done. ({time.time()-start_time:.2f}s)')

    def forward(self, acc, topol, f_emb, query):
        acc = acc.unsqueeze(1).type(torch.FloatTensor).to(self.device)
        query = [d.to(self.device) for d in query]
        q_emb = self.enc_q(query) 
        m_emb = self.enc_m(topol.to(self.device), f_emb.to(self.device))
        a_hat = self.predictor(q_emb, m_emb)
        lss = self.criterion(q_emb, m_emb)
        lss_mse = self.criterion_mse(a_hat, acc)
        return q_emb, m_emb, lss, lss_mse

    def evaluate(self):
        dataset, acc, topol, f_emb = next(iter(self.te_loader))
        with torch.no_grad():
            query = self.te_dataset.get_query(dataset)
            q, m, lss, lss_mse = self.forward(acc, topol, f_emb, query)
        recalls, medr, meanr = compute_recall(q.cpu(), m.cpu())
        return lss.item(), recalls, medr, meanr, lss_mse.item()

    def save_model(self, is_max=False, epoch=None, recall=None, medr=None, meanr=None, mse=None):
        print('==> saving models ... ')
        if is_max:
            fname = 'saved_model_max_recall.pt'
        else:
            fname = 'saved_model.pt'
        torch_save(self.args.check_pt_path, fname, {
            'enc_q': self.enc_q.cpu().state_dict(),
            'enc_m': self.enc_m.cpu().state_dict(),
            'predictor': self.predictor.cpu().state_dict(),
            'epoch': epoch,
            'recall': recall,
            'medr': medr,
            'meanr': meanr,
            'mse': mse
        })
        self.predictor.to(self.device)
        self.enc_q.to(self.device)
        self.enc_m.to(self.device)

    def save_scroes(self):
        f_write(self.args.log_path, f'cross_modal_space_results.txt', {
            'options': vars(self.args),
            'results': self.scores
        })

    def save_cross_modal_space(self):
        print('==> save the cross modal space from model-zoo ... ')
        self.tr_dataset, self.tr_loader = get_loader(self.args, mode='train')
        self.load_model_zoo()
        self.load_model_encoder()
        self.store_model_embeddings()

    def load_model_zoo(self):
        start_time = time.time()
        self.model_zoo = torch_load(self.args.model_zoo)
        print(f"==> {len(self.model_zoo['dataset'])} pairs have been loaded {time.time()-start_time:.2f}s")

    def load_model_encoder(self):
        print('==> loading model encoder ... ')
        loaded = torch_load(os.path.join(self.args.check_pt_path, 'saved_model_max_recall.pt'))
        self.enc_m = ModelEncoder(self.args).to(self.device)
        self.enc_m.load_state_dict(loaded['enc_m'])
        self.enc_m.eval()

    def store_model_embeddings(self):
        print('==> storing model embeddings ... ')
        start_time = time.time()
        embeddings = {'dataset': [],'m_emb': [],'topol': [],'acc': [],'n_params': []}
        
        for i, dataset in enumerate(self.model_zoo['dataset']): 
            emb_time = time.time()
            acc = self.model_zoo['acc'][i]
            n_params = self.model_zoo['n_params'][i]
            topol = self.model_zoo['topol'][i]
            f_emb = self.model_zoo['f_emb'][i]
            with torch.no_grad():
                m_emb = self.enc_m(
                    torch.Tensor(topol).unsqueeze(0).to(self.device), f_emb.unsqueeze(0).to(self.device))
            embeddings['dataset'].append(dataset) 
            embeddings['m_emb'].append(m_emb) 
            embeddings['topol'].append(topol) 
            embeddings['acc'].append(acc) 
            embeddings['n_params'].append(n_params) 

            if (i+1)%100 == 0:
                print(f'{i+1}th model done ({time.time()-emb_time:.2f})')

        torch_save(self.args.retrieval_path, 'retrieval.pt', embeddings)
        print(f'==> storing embeddings done. ({time.time()-start_time}s)')

    def test(self):
        print(f'Begin test process')
        start_time = time.time()
        self.init_loaders_for_meta_test()
        self.load_encoders_for_meta_test()
        self.load_cross_modal_space()
        self.meta_test()
        print(f'Process done ({time.time()-start_time:.2f})')

    def init_loaders_for_meta_test(self):
        print('==> loading meta-test loaders')
        self.tr_dataset, self.tr_loader = get_meta_test_loader(self.args, mode='train')
        self.te_dataset, self.te_loader = get_meta_test_loader(self.args, mode='test')

    def load_encoders_for_meta_test(self):
        print('==> loading encoders ... ')
        _loaded = torch_load(os.path.join(self.args.load_path, 'checkpoints', 'saved_model_max_recall.pt'))
        self.enc_q = QueryEncoder(self.args).to(self.device).eval()
        self.enc_q.load_state_dict(_loaded['enc_q'])
        self.predictor = PerformancePredictor(self.args).to(self.device).eval()
        self.predictor.load_state_dict(_loaded['predictor'])

    def load_cross_modal_space(self):
        print('==> loading the cross modal space ... ')
        self.cross_modal_info = torch_load(os.path.join(self.args.load_path, 'retrieval', 'retrieval.pt'))
        self.m_embs = torch.stack(self.cross_modal_info['m_emb']).to(self.device)
        
    def meta_test(self):
        print('==> meta-testing on unseen datasets ... ')
        
        for query_id, query_dataset in enumerate(self.tr_dataset.get_dataset_list()):
            self.tr_dataset.set_dataset(query_dataset)
            self.te_dataset.set_dataset(query_dataset)
            self.query_id = query_id
            self.query_dataset = query_dataset
            self.meta_test_results = {
                'query': self.query_dataset,
                'retrieval': {},
            }
            
            query = self.tr_dataset.get_query_set(self.query_dataset)
            query = torch.stack([d.to(self.device) for d in query])
            q_emb = self.get_query_embeddings(query).unsqueeze(0)
            retrieved = self.retrieve(q_emb, self.args.n_retrievals)

            mse_list = []
            score_list = []
            dataset_list = []
            acc_hat_list = []
            print(f' ========================================================================================================================')
            print(f' [query_id:{query_id+1}] query by {query_dataset} ... ')
            for k, retrieved_dataset in enumerate(retrieved['dataset']):
                topol = retrieved['topol'][k]
                npms = retrieved['n_params'][k]
                acc = retrieved['acc'][k]
                m_emb = retrieved['m_emb'][k].to(self.device)
                acc_hat = self.predictor(q_emb, m_emb).item()
                acc_hat_list.append(acc_hat)
                dataset_list.append(retrieved_dataset)

            top_0 = [0]
            top_j = []
            for j in [int(i) for i in np.argsort(acc_hat_list)[-3:]]:
                if not j == 0:
                    top_j.append(j)
            top_k_idx = top_0 + top_j
            
            print(' ========================================================================================================================')
            print(f' [query_id:{query_id+1}] fine-tuning on {query_dataset} ... ')
            for i, k in enumerate(top_k_idx):
                st = time.time()
                self.i = i
                self.k = k
                retrieved_dataset = retrieved['dataset'][k]
                topol = retrieved['topol'][k]
                npms = retrieved['n_params'][k]
                acc = retrieved['acc'][k]
                print(f' >>> [r_id:{k+1}({i+1})]: {retrieved_dataset}, n_parms: {npms}')
                self.meta_test_results['retrieval'][k] = {
                    'scores': {
                        'ep_lss': [],'ep_acc': [],
                        'acc': [],'acc_hat': acc_hat_list[k],
                        'ep_tr_time': [],'ep_te_time': [],
                    },
                    'info': {
                        'task': retrieved_dataset,
                        'npms': npms, #.item()
                        'topo': topol, #.tolist()
                    }
                }
                #####################################
                self.retrieved_dataset = retrieved_dataset
                self.retrieved_npms = npms
                self.model = self.get_model(self.retrieved_dataset, topol, self.tr_dataset.get_n_clss())
                self.lss_fn_meta_test = torch.nn.CrossEntropyLoss()
                self.optim = torch.optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=4e-5) 
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, float(self.args.n_eps_finetuning))
                lss, acc = self.fine_tune(k)
                mse = np.sqrt(np.mean((acc_hat-acc)**2))
                score_list.append(acc)
                mse_list.append(mse)
                self.save_meta_test_results(self.query_dataset)
                del self.model
                del self.optim
                del self.lss_fn_meta_test
                print(' ========================================================================================================================')

            self.save_meta_test_results(self.query_dataset)
            del self.meta_test_results
            del q_emb
            del retrieved

    def fine_tune(self, k):
        print(f' ==> finetuning {k}th retreival model ... ')
        self.curr_lr = self.args.lr
        for ep in range(self.args.n_eps_finetuning):
            self.curr_ep = ep
            running_loss = 0.0
            ep_tr_time = 0
            total = 0
            for b_id, batch in enumerate(self.tr_loader):    
                x, y = batch
                b_tr_lss, b_tr_time = self.fine_tune_step(x, y)
                running_loss += b_tr_lss * x.size(0)
                total += x.size(0)
                ep_tr_time += b_tr_time

            tr_lss = running_loss/len(self.tr_dataset) 
            te_lss, te_acc, ep_te_time = self.meta_test_evaluate()

            self.meta_test_results['retrieval'][k]['scores']['ep_tr_time'].append(ep_tr_time)
            self.meta_test_results['retrieval'][k]['scores']['ep_te_time'].append(ep_te_time)
            self.meta_test_results['retrieval'][k]['scores']['ep_acc'].append(te_acc)
            self.meta_test_results['retrieval'][k]['scores']['ep_lss'].append(te_lss)
            
            print(
                f' ==> [query_id:{self.query_id+1}]'+
                f'[r_id:{self.k+1}({self.i+1}):{self.retrieved_dataset}]'+
                f' ep:{ep+1}, tr_lss:{tr_lss:.3f},'+
                f' te_lss:{te_lss:.3f}, te_acc: {te_acc:.3f},'+
                f' tr_time:{ep_tr_time:.3f}s, te_time:{ep_te_time:.3f}s')
            
            self.save_meta_test_results(self.query_dataset)
            if (ep+1)%10==0:
                self.save_meta_test_model()
                print(f'model at ep: {ep} has been saved. ')
        
        return te_lss, te_acc

    def meta_test_evaluate(self):
        total = 0
        crrct = 0 
        ep_time = 0  
        running_loss = 0.0
        for b_id, batch in enumerate(self.te_loader):
            x, y = batch
            lss, y_hat, dura = self.meta_test_eval_step(x, y)
            ep_time += dura
            running_loss += lss.item() * x.size(0)
            total += y.size(0)
            _, y_hat = torch.max(y_hat.data, 1)
            crrct += (y_hat == y.to(self.device)).sum().item()
        ep_acc = crrct/total
        ep_lss = running_loss/len(self.te_dataset) # total 
        return ep_lss, ep_acc, ep_time
        
    def fine_tune_step(self, x, y):
        self.optim.zero_grad()
        self.model.train()
        st = time.time()
        y_hat = self.model(x.to(self.device))
        lss = self.lss_fn_meta_test(y_hat, y.to(self.device))
        lss.backward()
        self.optim.step()
        dura = time.time() - st
        self.scheduler.step()
        del x
        del y
        return lss.item(), dura

    def meta_test_eval_step(self, x, y):
        st = time.time()
        with torch.no_grad():
            self.model.eval()
            y_hat = self.model(x.to(self.device))
        lss = self.lss_fn_meta_test(y_hat, y.to(self.device))
        dura = time.time() - st
        del x
        del y
        return lss, y_hat, dura

    def get_query_embeddings(self, x_emb):
        print(' ==> converting dataset to query embedding ... ')
        q = self.enc_q(x_emb.unsqueeze(0)) 
        return q.squeeze()

    def get_model(self, task, topo, nclss):
        ks = topo[:20] 
        e = topo[20:40]
        d = topo[40:]
        return self.get_subnet(task, ks, e, d, nclss)

    def get_subnet(self, dataset, ks, e, d, nclss): 
        supernet = torch_load(os.path.join(self.args.model_zoo_raw, f'{dataset}.pt'))
        supernet.set_active_subnet(ks=ks, e=e, d=d)
        subnet = supernet.get_active_subnet(preserve_weight=True)
        subnet.classifier = torch.nn.Linear(1536, nclss)
        subnet = subnet.to(self.device)
        del supernet
        return subnet

    def retrieve(self, _q, n_retrieval):
        s_t = time.time()
        scores = torch.mm(_q, self.m_embs.squeeze().t()).squeeze()
        sorted, sorted_idx = torch.sort(scores, 0, descending=True)
        top_10_idx = sorted_idx[:n_retrieval]
        retrieved = {}
        for idx in top_10_idx:
            for k, v in self.cross_modal_info.items():
                if not k in retrieved:
                    retrieved[k] = []
                retrieved[k].append(v[idx])
        dura = time.time() - s_t
        self.meta_test_results['retrieval']['search_time'] = dura
        print(f'search time {dura:.5f} s')
        return retrieved

    def get_lr(self):
        for param_group in self.optim.param_groups:
            return param_group['lr']

    def save_meta_test_results(self, query_dataset):
        f_write(self.args.log_path, f'{query_dataset}.txt', {
            'options': vars(self.args),
            'results': self.meta_test_results
        })

    def save_meta_test_model(self):
        torch_save(
            self.args.check_pt_path, 
            f'{self.query_dataset}_{self.k}_{self.curr_ep}.pt', {
                'model': self.model,
                'curr_ep': self.curr_ep,
                'curr_lr': self.get_lr(), 
        })

