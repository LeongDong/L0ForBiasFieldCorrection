from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F

##################
#paper:The L0 Regularized Mumford–Shah Model for Bias Correction and Segmentation of Medical Images
#Author: Yuping Duan et al.
#IEEE Transactions on Image processing, 24(11), 2015.
#code: 2024.11.15
####################
imgpath = '../test98.png'
savepath = '.../smoothFFTorch.png'

def psf2otf(psf, outSize):
    nOps = 0
    psf_H, psf_W = psf.shape
    pad_H, pad_W = outSize[0] - psf_H, outSize[1] - psf_W
    psf_pad = F.pad(psf, pad=[0, pad_W, 0, pad_H], mode='constant', value=0)

    psf_pad = torch.roll(psf_pad, -int(psf_H / 2), 0)
    psf_pad = torch.roll(psf_pad, -int(psf_W / 2), 1)

    otf = torch.fft.fftn(psf_pad)

    nOps = nOps + psf_H * torch.log2(torch.Tensor([psf_H])) * 2 + psf_W * torch.log2(torch.Tensor([psf_W]))
    if(torch.max(torch.abs(torch.imag(otf))) / torch.max(torch.abs(otf)) <= nOps * torch.finfo(torch.float32).eps):
        otf = torch.real(otf)

    return otf

def GaussianKernel(size):

    sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    center = size // 2
    ele = (np.arange(size, dtype=np.float64) - center)

    kernel1d = np.exp(- (ele ** 2) / (2 * sigma ** 2)) #size * 1
    kernel = kernel1d[..., None] @ kernel1d[None, ...] #size * size
    kernel = torch.from_numpy(kernel)

    return kernel #size*size, Unnormalized gaussian kernel

def partialFilter(mask, size): #mask: B*C*H*W

    kernel = GaussianKernel(size)
    # kernel = kernel.cuda()
    pad = (size - 1) // 2
    maskPad = F.pad(mask, pad=[pad, pad, pad, pad], mode='constant') # (H+2pad)*(W+2pad)
    patches = maskPad.unfold(0, size, 1).unfold(1, size, 1) #H*W*size*size

    parKernels = patches * kernel.unsqueeze(0).unsqueeze(0) #(H*W*size*size)*(1*1*size*size)->H*W*size*size
    kernelNorm = parKernels / (parKernels.sum(dim=(-1,-2), keepdim=True) + 1e-9) #H*W*size*size
    kernelNorm = kernelNorm * mask.unsqueeze(-1).unsqueeze(-1) #(H*W*size*size) * (H*W*1*1)->H*W*size*size

    return kernelNorm

def convParFilt(image, kernel): #B*C*H*W

    size = kernel.shape[-1]
    pad = (size - 1) // 2
    imgPad = F.pad(image, pad=[pad, pad, pad, pad], mode='constant') #(H+2pad)*(W+2pad)
    imgPatches = imgPad.unfold(0, size, 1).unfold(1, size, 1) #H*W*size*size
    imgConv = kernel * imgPatches #H*W*size*size
    return imgConv.sum(dim=(-1, -2), keepdim=False) #H*W

def maskCreate(image): #1*1*H*W

    # image = image * 255
    # mask = torch.zeros_like(image)
    # mask[image > 20] = 1
    mask = torch.ones_like(image)
    return mask #H*W

def dx_forward(x): #forward difference along x-axis. x:H*W

    h1 = torch.diff(x, 1, 1)
    h2 = torch.reshape(x[:, 0], (x.shape[0], 1)) - torch.reshape(x[:, -1], (x.shape[0], 1))
    h = torch.hstack((h1, h2))

    return h

def dx_backward(x): #backward difference along x-axis

    h1 = torch.diff(x, 1, 1)
    h2 = torch.reshape(x[:, 0], (x.shape[0], 1)) - torch.reshape(x[:, -1], (x.shape[0], 1))
    h = torch.hstack((h2, h1))

    return h

def dy_forward(x): #forward difference along y-axis

    v1 = torch.diff(x, 1, 0)
    v2 = torch.reshape(x[0, :], (1, x.shape[1])) - torch.reshape(x[-1, :], (1, x.shape[1]))
    v = torch.vstack((v1, v2))

    return v

def dy_backward(x): #backward difference along y-axis

    v1 = torch.diff(x, 1, 0)
    v2 = torch.reshape(x[0, :], (1, x.shape[1])) - torch.reshape(x[-1, :], (1, x.shape[1]))
    v = torch.vstack((v2, v1))

    return v

def updateHV(u, lamda, beta):

    h = dx_forward(u)
    v = dy_forward(u)
    t = (h ** 2 + v ** 2) < lamda / beta
    h[t] = 0
    v[t] = 0

    return h, v

def updateU(h, v, b, parMask, imgFFT, lapOPE, beta):

    b = convParFilt(b, parMask)
    biasFFT = torch.fft.fft2(b)
    divOPE = dx_backward(h) + dy_backward(v)
    FFTu = (imgFFT - biasFFT - beta * torch.fft.fft2(divOPE)) / (1 + beta * lapOPE)
    u = torch.real(torch.fft.ifft2(FFTu))

    return u

def updateB(img, u, parMask, tau):

    Iu = img - u
    IuConv = convParFilt(Iu, parMask)
    b = IuConv / (1 + tau)

    return b

def L0BiasCorrection(img, lamda = 0.03, kappa = 1.01, tau = 0.01):
    betaMax = 1e7
    u = img
    b = img
    fx = torch.Tensor([[1, -1]])
    fy = torch.Tensor([[1], [-1]])

    mask = maskCreate(img)
    parMask = partialFilter(mask, size=21)
    # fxx = torch.Tensor([[1, -2, 1]])
    # fyy = torch.Tensor([[1],[-2],[1]])
    # fxy = torch.Tensor([[1, -1],[-1, 1]])

    M, N = img.shape
    outSize = torch.tensor([M, N])
    otfFx = psf2otf(fx, outSize)
    otfFy = psf2otf(fy, outSize)
    # otfFxx = psf2otf(fxx, outSize)
    # otfFyy = psf2otf(fyy, outSize)
    # otfFxy = psf2otf(fxy, outSize)

    imgFFT = torch.fft.fft2(img)
    lapOPE = torch.abs(otfFx) ** 2 + torch.abs(otfFy) ** 2 #laplacian operator
    # hesOPE = torch.abs(otfFxx) ** 2 + torch.abs(otfFyy) ** 2 + 2 * torch.abs(otfFxy) ** 2 #Hessian operator
    beta = 1

    while(beta < betaMax):
        h, v = updateHV(u, lamda, beta)
        u = updateU(h, v, b, parMask, imgFFT, lapOPE, beta)
        b = updateB(img, u, parMask, tau)
        beta = beta * kappa
    return b

if __name__=='__main__':

    Im = np.array(Image.open(imgpath), 'd')
    Im = torch.from_numpy(Im)
    Im = Im / 255.0
    minValue = 1e-9
    logIm = torch.log(Im + minValue)
    b = L0BiasCorrection(logIm)
    b = torch.exp(b)
    img = Im / (b + 1e-9)
    img = img.numpy()
    cv2.imwrite(savepath, img * 255)