def cut(data,x):#切片绘画

    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    img = np.squeeze(data)
    img = img.permute(1, 2, 0)

    # 选择一个切片（例如，选择第100个切片）
    img = img[:, :, x]

    # 绘制该切片
    plt.imshow(img.cpu().detach().numpy())
    plt.title("Cut"+x)
    plt.colorbar()
    plt.show()



def pca(data):#PCA降维

    #图像处理
    import matplotlib.pyplot as plt
    import numpy as np
    img = np.squeeze(data)
    img = img.permute(1, 2, 0)

    import torch
    from torchvision.transforms import ToPILImage
    from sklearn.decomposition import PCA

    #假设图像 shape 是 (H, W, C)，需要 reshape 为 (H*W, C)
    H, W, C= img.shape
    reshaped_image = img.reshape(-1, C)

    # 使用 PCA 将通道降到3
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(reshaped_image.cpu())

    # 还原回图像形状 (H, W, 3)
    reduced_image = reduced.reshape(H, W, 3)

    # 输出形状 (256, 256, 3)
    plt.title("PCA")
    plt.imshow(reduced_image)
    plt.show()



def pyvi(data):#多维绘画
    import numpy as np
    import pyvista as pv

    # 图像处理
    import matplotlib.pyplot as plt
    import numpy as np
    img = np.squeeze(data)
    img = img.permute(1, 2, 0)

    # 将数据转换为 PyVista 的网格格式
    img = img.cpu().detach().numpy()
    grid = pv.wrap(img)

    plotter = pv.Plotter()
    plotter.add_volume(grid)
    plotter.show()

    #z轴切片
    slice = grid.slice(normal="z")
    slice.plot()


def meanZ(data):#对z轴（第3维）进行求平均

    # 图像处理
    import matplotlib.pyplot as plt
    import numpy as np
    img = np.squeeze(data)
    img = img.permute(1, 2, 0)

    # 对z轴（第3维）进行求平均
    img = img.mean(dim=2)

    # 绘制降维后的数据
    plt.imshow(img.cpu().numpy())
    plt.title("Averaged data over z-axis")
    plt.colorbar()
    plt.show()


def sumZ(data,A,B):#对 z 轴上的数据求和

    # 图像处理
    import matplotlib.pyplot as plt
    import numpy as np
    img = np.squeeze(data)
    img = img.permute(1, 2, 0)

    img = img[:, :,A:B]

    # 对 z 轴上的数据求和，降维到 128x128
    img = img.sum(dim=2)

    # 绘制降维后的数据
    plt.imshow(img.cpu().detach().numpy())
    plt.title("Summed data over"+str(A)+" and "+str(B))
    plt.colorbar()
    plt.show()



def fromAtoB(data,A,B): #从A到B

    # 图像处理
    import matplotlib.pyplot as plt
    import numpy as np
    img = np.squeeze(data)
    img = img.permute(1, 2, 0)

    #从A到B
    plt.title("from"+str(A)+"to"+str(B))
    plt.imshow(img[:,:,A:B].cpu().detach().numpy())
    plt.show()


def pool(data):#使用平均池化降维

    # 图像处理
    import matplotlib.pyplot as plt
    import numpy as np
    img = np.squeeze(data)
    img = img.permute(1, 2, 0)

    import torch.nn.functional as F

    # 将数据转换为 (C, H, W) 形状，以便使用池化
    img = img.permute(2, 0, 1)  # 转换为 (190, 128, 128)

    # 使用2x2的平均池化进行降维
    pooled_data = F.avg_pool2d(img, kernel_size=2, stride=2)

    # 将池化后的数据转换回 (H, W, C) 形状
    pooled_data = pooled_data.permute(1, 2, 0)

    # 绘制池化后的数据
    plt.imshow(pooled_data[:, :, 0].cpu().detach().numpy())
    plt.title("Pooled data")
    plt.colorbar()
    plt.show()