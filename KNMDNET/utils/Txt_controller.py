import os
import shutil
class txt_writer():
    def __init__(self,file_dir,name):
        self.file_dir=file_dir
        if os.path.exists(self.file_dir)==False:
            os.makedirs(self.file_dir)
        self.name=name
        self.file_path=os.path.join(self.file_dir,self.name)
        self.clear_txt()
    def clear_txt(self):
        f=open(self.file_path,'w')
        f.close()
    def write_content(self,content):
        f=open(self.file_path,'a')
        for o in content:
            f.write(str(o))
            f.write(' ')
        f.close()
    def change_line(self):
        f=open(self.file_path,'a')
        f.write('\n')
        f.close()
    def write_line(self,content):
        self.write_content(content)
        self.change_line()
    def save_file(self,save_dir):
        shutil.copy(self.file_path,os.path.join(save_dir,self.name))



