from DataLoader.dataset.sources.vocsource import VocSource
import cv2
if __name__=='__main__':
    from setting.read_setting import config as cfg
    source = VocSource(data_dir=cfg.voc_data_dir,mode='test',img_size=None)
    num_image=len(source)
    for i in range(num_image):
        img, target, name, img_info, img_id=source[i]
        for t in target:
            cv2.rectangle(img,(int(t[1]),int(t[2])),(int(t[3]),int(t[4])),color=(0,255,255))
        cv2.imwrite(f"./imgs/{name}.png",img)