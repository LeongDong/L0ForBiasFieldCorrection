clear all;close all;

imgID = '/home/langdun/Downloads/HoL0MS/brain/test98N4.png'; 
Img = imread(imgID);
Img = Img(:,:,1);
Img = double(Img);
Im = log(Img+1);
[u, b] = L0L2Bias(Im, struct( 'lamda', 0.4, 'mu', 10, 'gamma', 1e-5, 'beta', 50,  'kappa', 1.01, 'betamax', 1e20));

uu = exp(u);
bb = exp(b);
imgc = Img ./ bb;
imgc = (imgc - min(min(imgc))) / (max(max(imgc)) - min(min(imgc)));
figure,imshow(imgc);hold on; axis off; axis equal;