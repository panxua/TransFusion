from matplotlib import pyplot as plt

def show_heatmap(hm,filepath="work_dirs/debugger/hm.png"):
    plt.imshow(hm,cmap=plt.cm.jet)
    plt.axis('off')
    plt.savefig(filepath)
    plt.close()

def show_img(img,filepath="work_dirs/debugger/img.png"):
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(filepath)
    plt.close()