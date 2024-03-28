import torch
from torch import nn
import torch.nn.functional as F
class NonLocakBlock(nn.Module):
    def __init__(self,channels:int,reduce_ratio:float=9.):
        super(NonLocakBlock,self).__init__()
        hidden_channels=int(channels//reduce_ratio)
        self.query_conv=nn.Conv2d(channels,hidden_channels,kernel_size=1)
        self.key_conv=nn.Conv2d(channels,hidden_channels,kernel_size=1)
        self.value_conv=nn.Conv2d(channels,channels,kernel_size=1)
        self.gamma=nn.Parameter(torch.zeros(1))
        self.softmax=nn.Softmax(dim=-1)
    def forward(self,x):
        batch_size,n_channels,height,width=x.shape
        query_x=self.query_conv(x).view(batch_size,-1,width*height).permute(0,2,1).contiguous()
        key_x=self.key_conv(x).view(batch_size,-1,width*height).contiguous()
        value_x=self.value_conv(x).view(batch_size,-1,width*height).contiguous()
        energy=torch.bmm(query_x,key_x)
        attention=self.softmax(energy)

        out=torch.bmm(value_x,attention.permute(0,2,1))
        out=out.view(batch_size,-1,height,width).contiguous()
        out=self.gamma*out+x
        return out
class GCA_Channel(nn.Module):
    def __init__(self,channels:int,out_size:int,reduce_ratio_nl:float,att_mode:str='origin'):
        super(GCA_Channel, self).__init__()
        assert att_mode in ['origin','post']
        self.attn_mode=att_mode
        if att_mode=='origin':
            self.pool=nn.AdaptiveMaxPool2d(output_size=out_size)
            self.non_local_att=NonLocakBlock(channels=channels,reduce_ratio=reduce_ratio_nl)
            self.sigmoid=nn.Sigmoid()
        elif att_mode=='post':
            self.pool =nn.AdaptiveMaxPool2d(output_size=out_size)
            self.non_local_att=NonLocakBlock(channels=channels,reduce_ratio=reduce_ratio_nl)
            self.conv_att=nn.Sequential(
                nn.Conv2d(channels,channels//4,kernel_size=1),
                nn.BatchNorm2d(channels//4),
                nn.ReLU(True),
                nn.Conv2d(channels//4,channels,kernel_size=1),
                nn.BatchNorm2d(channels),
                nn.Sigmoid()
            )
        else:
            raise NotImplementedError
    def forward(self,x):
        batch_size,num_channels,height,width=x.shape
        if self.attn_mode=='origin':
            gca=self.pool(x)
            gca=self.non_local_att(gca)
            gca=self.sigmoid(gca)
        elif self.attn_mode=='post':
            gca=self.pool(x)
            gca=self.non_local_att(gca)
            gca=self.conv_att(gca)
        else:
            raise NotImplementedError
        return gca

class GCA_Element(nn.Module):
    def __init__(self,channels,out_size,reduce_ratio_nl,attn_mode='origin'):
        super(GCA_Element, self).__init__()
        assert  attn_mode in ['origin','post']
        self.att_mode=attn_mode
        if attn_mode=='origin':
            self.pool=nn.AdaptiveMaxPool2d(output_size=out_size)
            self.non_local_att=NonLocakBlock(channels=channels,reduce_ratio=reduce_ratio_nl)
            self.sigmoid=nn.Sigmoid()
        elif attn_mode=='post':
            self.pool=nn.AdaptiveMaxPool2d(output_size=out_size)
            self.non_local_att=NonLocakBlock(channels,reduce_ratio=1)
            self.conv_att=nn.Sequential(
                nn.Conv2d(channels,channels//4,kernel_size=1),
                nn.BatchNorm2d(channels//4),
                nn.ReLU(True),
                nn.Conv2d(channels//4,channels,kernel_size=1),
                nn.BatchNorm2d(channels),
                nn.Sigmoid()
            )
        else:
            raise NotImplementedError
    def forward(self,x):
        if self.att_mode=='origin':
            gca=self.pool(x)
            gca=self.non_local_att(gca)
            gca=self.sigmoid(gca)
        elif self.att_mode=='post':
            gca=self.pool(x)
            gca=self.non_local_att(gca)
            gca=self.conv_att(gca)
        else:
            raise NotImplementedError
        return gca
class AGCB_Patch(nn.Module):
    def __init__(self,channels:int,out_size:int=2,reduce_ratio_nl:float=32.,att_mode='origin'):
        super(AGCB_Patch,self).__init__()
        self.out_size=out_size
        self.non_local=NonLocakBlock(channels=channels,reduce_ratio=reduce_ratio_nl)
        self.conv=nn.Sequential(
            nn.Conv2d(channels,channels,3,1,1),
            nn.BatchNorm2d(channels)
        )
        self.relu=nn.ReLU(True)
        self.attention=GCA_Channel(channels=channels,out_size=out_size,reduce_ratio_nl=reduce_ratio_nl,att_mode=att_mode)
        self.gamma=nn.Parameter(torch.zeros(1))
    def forward(self,x):
        gca=self.attention(x)
        batch_size,num_channels,height,width=x.shape
        local_x,local_y,attention_ind=[],[],[]
        step_h,step_w=height//self.out_size,width//self.out_size

        for i in range(self.out_size):
            for j in range(self.out_size):
                start_x,start_y=i*step_h,j*step_w
                end_x,end_y=min(start_x + step_h, height), min(start_y + step_w, width)
                if i==(self.out_size-1):
                    end_x=height
                if j==(self.out_size):
                    end_y=width
                local_x+=[start_x,end_x]
                local_y+=[start_y,end_y]
                attention_ind+=[i,j]
        index_cnt=2*self.out_size*self.out_size
        assert  len(local_x)==index_cnt
        context_list=[]
        for i in range(0, index_cnt, 2):
            block = x[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]
            attention = gca[:, :, attention_ind[i], attention_ind[i + 1]].view(batch_size, num_channels, 1, 1)
            context_list.append(self.non_local(block) * attention)
        tmp =[]
        for i in range(self.out_size):
            row_tmp=[]
            for j in range(self.out_size):
                row_tmp.append(context_list[j+i*self.out_size])
            tmp.append(torch.cat(row_tmp,3))
        context=torch.cat(tmp,2)

        context=self.conv(context)
        context=self.gamma*context+x
        context=self.relu(context)
        return context
class AGCB_Element(nn.Module):
    def __init__(self,channels:int,out_size:int=2,reduce_ratio_nl:float=32.,att_mode='origin'):
        super(AGCB_Element,self).__init__()
        self.out_size=out_size
        self.non_local=NonLocakBlock(channels=channels,reduce_ratio=reduce_ratio_nl)
        self.conv=nn.Sequential(
            nn.Conv2d(channels,channels,3,1,1),
            nn.BatchNorm2d(channels)
        )
        self.relu=nn.ReLU(True)
        self.attention=GCA_Element(channels=channels,out_size=out_size,reduce_ratio_nl=reduce_ratio_nl,attn_mode=att_mode)
        self.gamma=nn.Parameter(torch.zeros(1))
    def forward(self,x):
        gca=self.attention(x)
        batch_size,num_channels,height,width=x.shape
        local_x,local_y,attention_ind=[],[],[]
        step_h,step_w=height//self.out_size,width//self.out_size

        for i in range(self.out_size):
            for j in range(self.out_size):
                start_x,start_y=i*step_h,j*step_w
                end_x,end_y=min(start_x + step_h, height), min(start_y + step_w, width)
                if i==(self.out_size-1):
                    end_x=height
                if j==(self.out_size):
                    end_y=width
                local_x+=[start_x,end_x]
                local_y+=[start_y,end_y]
                attention_ind+=[i,j]
        index_cnt=2*self.out_size*self.out_size
        assert  len(local_x)==index_cnt
        context_list=[]
        for i in range(0, index_cnt, 2):
            block = x[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]
            # attention = gca[:, :, attention_ind[i], attention_ind[i + 1]].view(batch_size, num_channels, 1, 1)
            context_list.append(self.non_local(block))
        tmp =[]
        for i in range(self.out_size):
            row_tmp=[]
            for j in range(self.out_size):
                row_tmp.append(context_list[j+i*self.out_size])
            tmp.append(torch.cat(row_tmp,3))
        context=torch.cat(tmp,2)

        context=context*gca
        context=self.conv(context)
        context=self.gamma*context+x
        context=self.relu(context)
        return context

class AGCB_NoGCA(nn.Module):
    def __init__(self,channels:int,out_size:int=2,reduce_ratio_nl:float=32.):
        super(AGCB_NoGCA, self).__init__()
        self.out_size=out_size
        self.non_local=NonLocakBlock(channels,reduce_ratio=reduce_ratio_nl)
        self.conv=nn.Sequential(
            nn.Conv2d(channels,channels,3,1,1),
            nn.BatchNorm2d(channels)
        )
        self.relu=nn.ReLU(True)
        self.gamma=nn.Parameter(torch.zeros(1))
    def forward(self,x):
        batch_size,num_channels,height,width=x.shape
        local_x, local_y, attention_ind = [], [], []
        step_h, step_w = height // self.scale, width // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, height), min(start_y + step_w, width)
                if i == (self.scale - 1):
                    end_x = height
                if j == (self.scale - 1):
                    end_y = width

                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
                attention_ind += [i, j]
        index_cnt = 2 * self.scale * self.scale
        assert len(local_x) == index_cnt

        context_list = []
        for i in range(0, index_cnt, 2):
            block = x[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]
            context_list.append(self.non_local(block))
        tmp = []
        for i in range(self.scale):
            row_tmp = []
            for j in range(self.scale):
                row_tmp.append(context_list[j + i * self.scale])
            tmp.append(torch.cat(row_tmp, 3))
        context = torch.cat(tmp, 2)

        context = self.conv(context)
        context = self.gamma * context + x
        context = self.relu(context)
        return context
class CPM(nn.Module):
    def __init__(self, planes, block_type, scales=(3,5,6,10), reduce_ratios=(4,8), att_mode='origin'):
        super(CPM, self).__init__()
        assert block_type in ['patch', 'element']
        assert att_mode in ['origin', 'post']

        inter_planes = planes // reduce_ratios[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(planes, inter_planes, kernel_size=1),
            nn.BatchNorm2d(inter_planes),
            nn.ReLU(True),
        )

        if block_type == 'patch':
            self.scale_list = nn.ModuleList(
                [AGCB_Patch(inter_planes, scale=scale, reduce_ratio_nl=reduce_ratios[1], att_mode=att_mode)
                 for scale in scales])
        elif block_type == 'element':
            self.scale_list = nn.ModuleList(
                [AGCB_Element(inter_planes, scale=scale, reduce_ratio_nl=reduce_ratios[1], att_mode=att_mode)
                 for scale in scales])
        else:
            raise NotImplementedError

        channels = inter_planes * (len(scales) + 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, planes, 1),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        reduced = self.conv1(x)

        blocks = []
        for i in range(len(self.scale_list)):
            blocks.append(self.scale_list[i](reduced))
        out = torch.cat(blocks, 1)
        out = torch.cat((reduced, out), 1)
        out = self.conv2(out)
        return out