function [u, b] = L0Bias(img, options)

lamda = options.lamda;
mu = options.mu;
tau = options.tau;
beta = options.beta;
kappa = options.kappa;
betamax = options.betamax;

[M, N] = size(img);
u = img;
b = zeros(M,N);

fx = [1, -1];
fy = [1; -1];
fxx = [1, -2, 1];
fyy = [1; -2; 1];
fxy = [1, -1; -1, 1];


otfFx = psf2otf(fx, [M,N]);
otfFy = psf2otf(fy, [M,N]);
otfFxx = psf2otf(fxx, [M,N]);
otfFyy = psf2otf(fyy, [M,N]);
otfFxy = psf2otf(fxy, [M,N]);

bx = dx_forward(b);
by = dy_forward(b);
bxx = dx_backward(bx);
byy = dy_backward(by);
bxy = dy_forward(bx);

qxx = bxx;
qxy = bxy;
qyy = byy;

h = dx_forward(u);
v = dy_forward(u);

lapOPE = abs(otfFx).^2 + abs(otfFy).^2;
hesOPE = abs(otfFxx).^2 + abs(otfFyy).^2 + 2 * abs(otfFxy).^2;

while(beta < betamax)
    
    b = updateB(img, u, hesOPE, beta, tau, qxx, qxy, qyy);
    u = updateU(h, v, b, img, lapOPE, beta);
    [h, v] = updateHV(u, lamda, beta);
    
    bx = dx_forward(b);
    by = dy_forward(b);
    bxx = dx_backward(bx);
    byy = dy_backward(by);
    bxy = dy_forward(bx);
    
    qxx = sign(bxx).*max(abs(bxx) - mu / beta, 0);
    qxy = sign(bxy).*max(abs(bxy) - mu / beta, 0);
    qyy = sign(byy).*max(abs(byy) - mu / beta, 0);
    
    beta = kappa * beta;
end

function [h, v] = updateHV(u, lamda, beta)

h = dx_forward(u);
v = dy_forward(u);

t = (h.^2 + v.^2) < 2 * lamda / beta;
h(t) = 0;
v(t) = 0;

function [u] = updateU(h, v, b, img, lapOPE, beta)

hx = dx_backward(h);
vy = dy_backward(v);
Normin = fft2(img - b) - beta * (fft2(hx) + fft2(vy));
Denormin = 1 + beta * lapOPE;
FFTu = Normin ./ Denormin;
u = real(ifft2(FFTu));

function [b] = updateB(img, u, hesOPE, beta, tau, qxx, qxy, qyy)

qxxxx = dx_forward(dx_backward(qxx));
qxyxy = dy_backward(dx_backward(qxy));
qyxyx = qxyxy;
qyyyy = dy_forward(dy_backward(qyy));
Normin = fft2(img - u) + beta * (fft2(qxxxx) + fft2(qxyxy) + fft2(qyxyx) + fft2(qyyyy));
Denorm = 1 + tau + beta * hesOPE;
biasFFT = Normin ./ Denorm;
b = real(ifft2(biasFFT));

function [h] = dx_forward(x)
h1 = diff(x,1,2);
h2 = x(:,1) - x(:,end);
h = [h1, h2];

function [h] = dx_backward(x)
h1 = diff(x,1,2);
h2 = x(:,1) - x(:,end);
h = [h2, h1];

function [v] = dy_forward(x)
v1 = diff(x,1,1);
v2 = x(1,:) - x(end,:);
v = [v1;v2];

function [v] = dy_backward(x)
v1 = diff(x,1,1);
v2 = x(1,:) - x(end,:);
v = [v2;v1];




