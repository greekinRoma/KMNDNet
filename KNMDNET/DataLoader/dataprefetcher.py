import torch
class DataPrefetcher:
    def __init__(self,loader,use_cuda:bool=True):
        self.loader=iter(loader)
        self.use_cuda=use_cuda
        if use_cuda:
            self.stream=torch.cuda.Stream()
        else:
            self.stream=torch.Stream()
        self.input_cuda=self._input_cuda_for_image
        self.record_stream=DataPrefetcher._record_stream_for_image
    def preload(self):
        try:
            self.next_input,self.next_masks,self.next_use_augs,self.next_target,_=next(self.loader)
        except StopIteration:
            self.next_input=None
            self.next_target=None
            self.next_masks=None
            self.next_use_augs=None
            return
        with torch.cuda.stream(self.stream):
            self.input_cuda()
            #self.next_target=self.next_target.cuda(non_blocking=True)
    def next(self):
        if self.use_cuda:
            torch.cuda.current_stream().wait_stream(self.stream)
        input=self.next_input
        target=self.next_target
        masks=self.next_masks
        use_augs=self.next_use_augs
        if self.use_cuda:
            if input is not None:
                self.record_stream(input)
            if target is not None:
                target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input,masks,use_augs,target

    def _input_cuda_for_image(self):
        if self.use_cuda:
            self.next_input=self.next_input.cuda(non_blocking=True)
    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())