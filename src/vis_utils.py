from matplotlib import pyplot as plt
import torch
import os

def show_heatmap(hm,filepath="work_dirs/debugger/hm.png", points=None, bboxes=None):
    plt.imshow(hm,cmap=plt.cm.jet)
    # plt.axis('off')
    if points is not None:
        for x,y in points:
            plt.plot(x,y,"ko", markersize=2)
    if bboxes is not None:
        plot_bboxes(bboxes, "black", 0.5)
    plt.savefig(filepath)
    plt.close()

def show_img(img,filepath="work_dirs/debugger/img.png"):
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(filepath)
    plt.close()
def plot_bboxes(bboxes, color, alpha):
    for bbox in bboxes:
        plot_face(bbox[:4], color, alpha)
        plot_face(bbox[4:], color, alpha)
        plot_face(torch.stack([bbox[0],bbox[1],bbox[5],bbox[4]], dim=0), color, alpha)
        plot_face(torch.stack([bbox[1],bbox[2],bbox[6],bbox[5]], dim=0), color, alpha)
        plot_face(torch.stack([bbox[2],bbox[3],bbox[7],bbox[6]], dim=0), color, alpha)
        plot_face(torch.stack([bbox[0],bbox[3],bbox[7],bbox[4]], dim=0), color, alpha)
def plot_face(corners, color, alpha):
    print(corners)
    xs = corners[:,0]
    ys = corners[:,1]
    plt.fill(xs,ys,color=color,alpha=alpha)

