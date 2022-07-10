import scipy.sparse as sp
import numpy as np
import argparse
import os
from torch.autograd import Variable
from target_model import *
from gcn import *
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description="Run.")
parser.add_argument('--dataset',default='ogb_arxiv', help='ogb_arxiv or reddit')
parser.add_argument('--gpu',type=int,default=0,help='id of gpu')
parser.add_argument('--lr',type=float,default=0.1,help="optimization step size")
parser.add_argument('--n_iter',type=int,default=20, help='number of iteration of gradient optimization')
parser.add_argument('--n_pop',type=int,default=50, help='population of each NES iteration')
parser.add_argument('--sigma',type=float,default=0.01, help='sigma of NES')
parser.add_argument('--query',type=int,default=1, help='whether to use query-based or instead white-box attack.')
parser.add_argument('--targeted',type=int,default=0, help='to perform targeted or untargeted attack')
parser.add_argument('--load_dir',type=str,default='.', help='dir of datasets')
parser.add_argument('--n_victim',type=int,default=12, help='number of victim nodes.')
parser.add_argument('--n_fake',type=int,default=4, help='number of added fake nodes.')
args=parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

N_iter = args.n_iter
N_pop = args.n_pop
load_dir = args.load_dir

if args.dataset=="reddit":
    adj=sp.load_npz(load_dir+"/datasets/reddit/reddit_adj.npz")
    adj=adj+adj.transpose()
    data=np.load(load_dir+"/datasets/reddit/reddit.npz")
    features=data['feats']
    features[:,:2]*=0.025 # scale them to usual range
    trainlabels=data['y_train']
    vallabels=data['y_val']
    testlabels=data['y_test']
    train_index=data['train_index']
    val_index=data['val_index']
    test_index=data['test_index']
    max_lim = 0.25
if args.dataset=="ogb_arxiv":
    dir=load_dir+"/datasets/arxiv"
    adj,features,labels,train_index,val_index,test_index=loadogb('arxiv',dir)
    trainlabels=labels[train_index]
    vallabels=labels[val_index]
    testlabels=labels[test_index]
    max_lim = 1.0

num_classes=np.max(trainlabels)+1
num_features=features.shape[1]

model=gcn(num_features,num_classes).cuda()
model.eval()

if args.dataset=="reddit":
    model.load_state_dict(torch.load('models/gcn_reddit'))
if args.dataset=="ogb_arxiv":
    model.load_state_dict(torch.load('models/gcn_ogb_arxiv'))

model=model.cuda()
model.eval()

processed_adj=GCNadj(adj)
sparserow=torch.LongTensor(processed_adj.row).unsqueeze(1)
sparsecol=torch.LongTensor(processed_adj.col).unsqueeze(1)
sparseconcat=torch.cat((sparserow,sparsecol),1).cuda()
sparsedata=torch.FloatTensor(processed_adj.data).cuda()
adjtensor=torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(processed_adj.shape)).cuda()

featuretens=torch.FloatTensor(features)
featuretensor=Variable(featuretens,requires_grad=True).cuda()

out=model(featuretensor,adjtensor,dropout=0)
testout=out[test_index]
testoc=testout.argmax(1)
test_acc=torch.eq(testoc.cpu(),torch.tensor(testlabels)).sum()
test_acc=test_acc/(len(testlabels)+0.0)
print("Acc of clean model:", test_acc)

testo=testoc.data

# ground truth labels predicted by model
ground_truth_label = out.argmax(1)
target_label = (ground_truth_label+2)%num_classes

def CWLoss(score, t_label):
    '''
    score = [
        [0.1, 0.4, ..., -0.1],
        ...,
        [0.1, 0.4, ..., -0.1]
    ]
    t_label = [1, 3, 0, ..., 1] target label or ground truth label
    '''
    
    one_hot = torch.zeros(t_label.shape[0], num_classes).cuda().scatter_(1, t_label.unsqueeze(1), 1) * (10**10)
    score_copy = score.clone().detach()
    score_copy = score_copy - one_hot

    if args.targeted:
        mask = (score_copy.max(1)[0] - torch.gather(score, dim=1, index=t_label.unsqueeze(1)).squeeze(1)).ge(0).float()
        cw_loss = ((score_copy.max(1)[0] - torch.gather(score, dim=1, index=t_label.unsqueeze(1)).squeeze(1)) * mask).sum(0)
    else:
        mask = (torch.gather(score, dim=1, index=t_label.unsqueeze(1)).squeeze(1) - score_copy.max(1)[0]).ge(0).float()
        cw_loss = ((torch.gather(score, dim=1, index=t_label.unsqueeze(1)).squeeze(1) - score_copy.max(1)[0]) * mask).sum(0)
    return cw_loss

def compute_AV(adj, featuretens, victim_node_id, fake_node_id):

    # Add one Edge
    add1=sp.csr_matrix((1,adj.shape[0]))
    add2=sp.csr_matrix((adj.shape[0]+1,1))
    adj=sp.vstack([adj,add1])
    adj=sp.hstack([adj,add2])
    adj.row=np.hstack([adj.row, [victim_node_id, fake_node_id]])
    adj.col=np.hstack([adj.col, [fake_node_id, victim_node_id]])
    adj.data=np.hstack([adj.data,[1.0,1.0]])

    processed_adj=GCNadj(adj)
    sparserow=torch.LongTensor(processed_adj.row).unsqueeze(1)
    sparsecol=torch.LongTensor(processed_adj.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow,sparsecol),1).cuda()
    sparsedata=torch.FloatTensor(processed_adj.data).cuda()
    adjtensor=torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(processed_adj.shape)).cuda()

    if not args.query:
        features_added = torch.zeros([1, num_features],requires_grad=True)
        optimizer=torch.optim.Adam([{'params':[features_added]}], lr=args.lr)
    else:
        features_added = torch.zeros([1, num_features],requires_grad=True).cuda()

    for epoch in range(N_iter//2):
        if not args.query: # white box attack
            optimizer.zero_grad()
            model.eval()
            
            features_all = torch.cat([featuretens, features_added.clamp(-max_lim,max_lim)]).cuda()
            out1=model(features_all,adjtensor,dropout=0)
            testout1=out1[[victim_node_id]]       
            if not args.targeted:
                l2 = CWLoss(testout1,ground_truth_label[[victim_node_id]])
            else:
                l2 = CWLoss(testout1,target_label[[victim_node_id]])
            l2.backward()
            optimizer.step()
            print("t",l2)
        else:
            with torch.no_grad():
                g = torch.zeros(num_features).cuda()
                for i in range(N_pop):
                    noise = torch.normal(torch.zeros(num_features), torch.ones(num_features)).cuda()

                    features_all = torch.cat([featuretens.cuda(), (features_added+noise*args.sigma).clamp(-max_lim,max_lim)]).cuda()
                    out1=model(features_all,adjtensor,dropout=0)
                    testout1=out1[[victim_node_id]]       
                    if not args.targeted:
                        l2 = CWLoss(testout1,ground_truth_label[[victim_node_id]])
                    else:
                        l2 = CWLoss(testout1,target_label[[victim_node_id]])
                    g += noise*l2

                    features_all = torch.cat([featuretens.cuda(), (features_added-noise*args.sigma).clamp(-max_lim,max_lim)]).cuda()
                    out1=model(features_all,adjtensor,dropout=0)
                    testout1=out1[[victim_node_id]]       
                    if not args.targeted:
                        l2 = CWLoss(testout1,ground_truth_label[[victim_node_id]])
                    else:
                        l2 = CWLoss(testout1,target_label[[victim_node_id]])
                    g -= noise*l2            
                g /= (2*N_pop*args.sigma)

                '''
                In our paper, we use Projected Gradient Descent to compute Adversarial Vulnerability. However, in large
                network the computation cost may be intolerable. Here, we use Fast Gradient Sign Method (Goodfellow et, al., 2014)
                instead. FGSM saves a lot time (~10x faster) and makes much fewer queries to the victim model, while it suffers from
                degration of accuracy in estimating the Adversarial Vulnerability.
                '''
                features_added = ((g.ge(0).float().unsqueeze(0))*(2*max_lim)-max_lim)
                break

    return features_added.data[0]

def cluster_attack(adj, featuretens, N_victim, N_fake):
    node_list = list(range(adj.shape[0]))
    random.shuffle(node_list)
    victim_node_list = node_list[:N_victim] # randomly generate N_victim fake nodes

    fake_node_list = list(range(adj.shape[0],adj.shape[0]+N_fake))

    # compute adversarial vulnerability
    AVs = {}
    for victim_node_id in victim_node_list:
        print("Computing Adversarial Vulnerability for victim node", victim_node_id)
        AVs[victim_node_id] = compute_AV(adj, featuretens, victim_node_id, adj.shape[0])

    # cluster the victim nodes
    estimator = KMeans(n_clusters=N_fake)
    MDFs = np.zeros((N_victim, num_features))
    for i, victim_node in enumerate(victim_node_list):
        MDFs[i] = AVs[victim_node].cpu().numpy()
    estimator.fit(MDFs)

    node_to_cluster = {} # Mapping of each victim node to which cluster
    cluster_dist = [0 for _ in range(N_fake)] # the size of each cluster
    label_pred = estimator.labels_
    for i, victim_node in enumerate(victim_node_list):
        node_to_cluster[victim_node] = fake_node_list[label_pred[i]]
        cluster_dist[label_pred[i]]+=1
    print(cluster_dist)
    
    # add fake nodes to original graph
    add1=sp.csr_matrix((N_fake,adj.shape[0]))
    add2=sp.csr_matrix((adj.shape[0]+N_fake,N_fake))
    adj=sp.vstack([adj,add1])
    adj=sp.hstack([adj,add2])
    newedgesx=[]
    newedgesy=[]
    newdata=[]
    for victim_node in victim_node_list:
        newedgesx.extend([victim_node, node_to_cluster[victim_node]])
        newedgesy.extend([node_to_cluster[victim_node], victim_node])
        newdata.extend([1.,1.])
    adj.row=np.hstack([adj.row, newedgesx])
    adj.col=np.hstack([adj.col, newedgesy])
    adj.data=np.hstack([adj.data, newdata])    

    # normalize adj
    processed_adj=GCNadj(adj)
    sparserow=torch.LongTensor(processed_adj.row).unsqueeze(1)
    sparsecol=torch.LongTensor(processed_adj.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow,sparsecol),1).cuda()
    sparsedata=torch.FloatTensor(processed_adj.data).cuda()
    adjtensor=torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(processed_adj.shape)).cuda()

    if not args.query:
        features_added = torch.zeros([N_fake, num_features],requires_grad=True)
        optimizer=torch.optim.Adam([{'params':[features_added]}], lr=args.lr)
    else:
        features_added = torch.zeros([N_fake, num_features],requires_grad=True).cuda()

    for epoch in range(N_iter):
        if not args.query:
            model.eval()
            features_all = torch.cat([featuretens, features_added.clamp(-max_lim,max_lim)]).cuda()
            out1=model(features_all,adjtensor,dropout=0)
            testout1=out1[victim_node_list]
            if not args.targeted:
                l2 = CWLoss(testout1,ground_truth_label[victim_node_list])
            else:
                l2 = CWLoss(testout1,target_label[victim_node_list])
            optimizer.zero_grad()
            l2.backward()
            optimizer.step()
            print(l2)
        else:
            with torch.no_grad():
                g = torch.zeros([N_fake, num_features]).cuda()
                for i in range(N_pop):
                    noise = torch.normal(torch.zeros([N_fake, num_features]), torch.ones([N_fake, num_features])).cuda()

                    features_all = torch.cat([featuretens.cuda(), (features_added+noise*args.sigma).clamp(-max_lim,max_lim)]).cuda()
                    out1=model(features_all,adjtensor,dropout=0)
                    testout1=out1[victim_node_list]       
                    if not args.targeted:
                        l2 = CWLoss(testout1,ground_truth_label[victim_node_list])
                    else:
                        l2 = CWLoss(testout1,target_label[victim_node_list])
                    g += noise*l2

                    features_all = torch.cat([featuretens.cuda(), (features_added-noise*args.sigma).clamp(-max_lim,max_lim)]).cuda()
                    out1=model(features_all,adjtensor,dropout=0)
                    testout1=out1[victim_node_list]       
                    if not args.targeted:
                        l2 = CWLoss(testout1,ground_truth_label[victim_node_list])
                    else:
                        l2 = CWLoss(testout1,target_label[victim_node_list])
                    g -= noise*l2            
                g /= (2*N_pop*args.sigma)

                features_added -= args.lr * g
                features_added.clamp_(-max_lim, max_lim)

                # only used for print loss
                features_all = torch.cat([featuretens.cuda(), (features_added).clamp(-max_lim,max_lim)]).cuda()
                out1=model(features_all,adjtensor,dropout=0)
                testout1=out1[[victim_node_list]]                       
                if not args.targeted:
                    l2 = CWLoss(testout1,ground_truth_label[[victim_node_list]])
                else:
                    l2 = CWLoss(testout1,target_label[[victim_node_list]])
                print("loss = ", l2, "avg norm L1 = ", features_added.abs().mean())

    features_all = torch.cat([featuretens.cuda(), (features_added).clamp(-max_lim,max_lim)]).cuda()
    out1=model(features_all,adjtensor,dropout=0)
    if not args.targeted:
        succ_acc=torch.eq(out1[victim_node_list].argmax(1).cpu(), ground_truth_label[victim_node_list].cpu()).sum()
        succ_acc=1-succ_acc/(len(victim_node_list)+0.0)
    else:
        succ_acc=torch.eq(out1[victim_node_list].argmax(1).cpu(), target_label[victim_node_list].cpu()).sum()
        succ_acc=succ_acc/(len(victim_node_list)+0.0)
    return succ_acc.data

N_victim=args.n_victim
N_fake=args.n_fake
N_round=100 # run 100 times

sum = 0
for i in range(N_round):
    succ_acc = cluster_attack(adj, featuretens, N_victim, N_fake)
    sum += succ_acc
    print("Round ["+str(i)+"], Using ["+str(N_fake)+"] fake nodes to attack ["+str(N_victim)+"] nodes, succ rate = "+str(succ_acc)+", total succ rate = "+str(sum/(i+1)))