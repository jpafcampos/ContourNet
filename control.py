import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.Net import Net
from args import args, flags
from data.vocdataset import dataset
from torch.utils.data import DataLoader
import time
from PIL import Image
import os
import torchvision
from tensorboardX import SummaryWriter
from thop import profile
from torchvision import transforms

writer = SummaryWriter('./tensorboard_logs/iter1')

class control():
    net = Net()
    #writer.add_graph(net, input_to_model=torch.rand(1, 3, 224, 224))
    #flops, params = profile(net, input_size=(1, 3, 224, 224))
    #print('flops = ', flops/1024/1024/1024, 'Gflops,   params = ', params/1024/1024, 'M')
    def __init__(self):
        pass

    def compute_loss(self, prediction, label):
        label = label.long()
        mask = label.float()
        num_positive = torch.sum((mask==1).float()).float()
        num_negative = torch.sum((mask==0).float()).float()

        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
        mask[mask == 2] = 0
        cost = torch.nn.functional.binary_cross_entropy(
                prediction.float(),label.float(), weight=mask, reduce=False)
        return torch.sum(cost)

    def train(self):
        data = dataset(flags.file_root,
                       flags.base_root_img,
                       flags.base_root_mask,
                       flags.mode)
        dataloader = DataLoader(data,
                                batch_size=2,
                                shuffle=True,
                                num_workers=8)
        #val_data = dataset('/home/joao/Documents/CNET/pascal/PASCALContourData/val.txt',
        val_data = dataset('/content/val.txt',
                           flags.base_root_img,
                           flags.base_root_mask,
                           flags.mode)
        val_dataloader = DataLoader(val_data,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=4)

        #small_lr_layers = list(map(id, self.net.model.parameters()))
        #large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.net.parameters())
        #self.net.load_state_dict(torch.load('./checkpoint/iter2.pth'))
        optimizer = torch.optim.SGD(self.net.parameters(), lr=1e-3, momentum=0.9)
        #optimizer = torch.optim.Adam(params=self.net.parameters(),lr=1e-3) #no-weight-decay
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True,patience=2, factor=0.2,mode='min')

        self.net.to(flags.device)
        self.net.train()
        for epoch in range(flags.epoches):
            running_loss = 0
            start_time = time.time()
            self.net.train()
            for i, (x, y, img_name) in enumerate(dataloader):
                img_name = str(img_name).split('/')[-1].split('.')[0]
                torch.cuda.empty_cache()
                x = x.to(flags.device)
                y = y.to(flags.device)

                results = self.net(x)
                print("results type")
                print(type(results))
                optimizer.zero_grad()
                loss = torch.zeros(1).to(flags.device)
                for r in results:
                    loss = loss + cross_entropy_loss_RCF(r, y)
                counter += 1
                loss = loss / 10
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                #print('epoch: ', epoch, 'step: ', i, 'loss = ', loss.item())
            torch.cuda.empty_cache()
            self.net.eval()
            val_loss = 0
            for i, (x, y, img_name) in enumerate(val_dataloader):
                img_name = str(img_name).split('/')[-1].split('.')[0]
                torch.cuda.empty_cache()
                x = x.to(flags.device)
                y = y.to(flags.device)

                results = self.net(x)
                val_loss += self.compute_loss(results, y).item()

            scheduler.step(val_loss, epoch=epoch)

            end_time = time.time()
            print('epoch: ', epoch, 'total_loss = ', running_loss, 'time = ', end_time-start_time)
            writer.add_scalar('epoch_loss', running_loss, epoch)

            #add-wight-histogram
            #for i, (name, param) in enumerate(self.net.named_parameters()):
            #    if 'bn' not in name:
            #        writer.add_histogram(name, param, epoch)

        #save-checkpoint
        #name = time.strftime('%mm_%dd_%Hh_%Mm_%Ss', time.localtime(time.time()))
        name = flags.checkpoint_name
        torch.save(self.net.state_dict(), './checkpoint/' + name + '.pth')

    def predict(self):
        img_list = []
        data = dataset('/content/val.txt',
                       flags.base_root_img,
                       flags.base_root_mask,
                       flags.mode)
        dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=8)
        self.net.load_state_dict(torch.load('./checkpoint/'+flags.checkpoint_name+'.pth'))

        self.net.eval()
        self.net.to(flags.device)

        for i, (x, y, img_name) in enumerate(dataloader):
            torch.cuda.empty_cache()
            x = x.to(flags.device)
            y = y.to(flags.device)
            result, layer_1, layer_2, layer_3, layer_4 = self.net(x)
            result = torch.squeeze(result, dim=0)
            result = torch.squeeze(result, dim=0)
            layer_1 = layer_1.permute(1, 0, 2, 3)
            layer_2 = layer_2.permute(1, 0, 2, 3)
            layer_3 = layer_3.permute(1, 0, 2, 3)
            layer_4 = layer_4.permute(1, 0, 2, 3)

            torchvision.utils.save_image(layer_1,'./result/'+str(i)+'_'+'layer_1.png')
            torchvision.utils.save_image(layer_2,'./result/'+str(i)+'_'+'layer_2.png')
            torchvision.utils.save_image(layer_3,'./result/'+str(i)+'_'+'layer_3.png')
            torchvision.utils.save_image(layer_4,'./result/'+str(i)+'_'+'layer_4.png')

            y = torch.squeeze(y, dim=0)
            #image = torchvision.utils.make_grid([result, y], padding=2)
            name = str(img_name).split('/')[-1].split('.')[0]
            img = result.cpu().detach().numpy()*255
            img = Image.fromarray(np.array(img))
            img = img.convert('L')
            img.save('content/'+str(name)+'.png')
            #writer.add_image(img_name, image)
            #result = torch.squeeze(result, dim=0)
            #result = torch.squeeze(result, dim=0)
            #result = Image.fromarray(result.cpu().detach().numpy()*255.0)
            #result = result.convert('L')
            #name = time.strftime('%mm_%dd_%Hh', time.localtime(time.time()))
            #img_name = str(img_name).split('/')[-1].split('.')[0]
            #dir = './result/'+ name +'/'+flags.mode+'/'
            #if not os.path.exists(dir):
            #    os.makedirs(dir)
            #result.save(dir + img_name + '.jpg')

    def predict_oneshot(self, filename):
        name = filename.split('/')[-1].split('.')[0]
        img_path = filename
        #gt_path = '/home/zhangcb/Desktop/VOCpreprocessed/PASCALContourData/groundTruth_val/' + name+ '.png'
        res_path = './result/' + name + '.png'

        img = np.array(Image.open(img_path))
        toTensorOp = transforms.ToTensor()
        img = toTensorOp(img)
        img = img.unsqueeze(0)
        img = img.to('cuda:0')
        print(img.shape)

        self.net.load_state_dict(torch.load('./checkpoint/iter3.pth'))

        self.net.eval()
        self.net.to(flags.device)

        with torch.no_grad():
            result = self.net(img).cpu().numpy()*255
            print(result.shape)
            result = result.squeeze(0)
            result = result.squeeze(0)
            result = Image.fromarray(np.array(result))
            result = result.convert('L')
            result.save(res_path)





