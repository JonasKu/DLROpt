using LinearAlgebra
using PyPlot

function f1(x::Array{Float64,2},y::Array{Float64,1})
    return 0.5*norm(x - y*y',2).^2;
end

function df1(x::Array{Float64,2},y::Array{Float64,1})
    return x-y*y';
end

function f2(x::Array{Float64,2},y::Array{Float64,1})
    return 0.5*norm(x - y*y',2).^2;
end

function df2(x::Array{Float64,2},y::Array{Float64,1})
    return x-y*y';
end

N = 20;
x = zeros(N,N);
y = rand(N);

f = x-> f1(x,y);
df = x-> df1(x,y);

alpha = 1e-1;
eps = 1e-5;

sdHistory = [f(x)];

# steepest descent
while f(x) > eps
    x .= x - alpha*df(x);
    push!(sdHistory, f(x))
    println(f(x))
end

############# DLR #############
r = 1;
x = zeros(N,N);
# Low-rank approx of init data:
X,S,W = svd(x); 
    
# rank-r truncation:
X = X[:,1:r]; 
W = W[:,1:r];
S = Diagonal(S);
S = S[1:r, 1:r]; 

K = zeros(N,r);
L = zeros(N,r);

DLRHistory = [f(X*S*W')];

# unconventional integrator
while f(X*S*W') > eps

    gradient = df(X*S*W');

    ###### K-step ######
    K .= X*S;

    K .= K .- alpha*gradient*W;

    XNew,STmp = qr(K); # optimize bei choosing XFull, SFull
    XNew = XNew[:, 1:r]; 

    MUp = XNew' * X;

    ###### L-step ######
    L = W*S';

    L .= L .- alpha*(X'*gradient)';
            
    WNew,STmp = qr(L);
    WNew = WNew[:, 1:r]; 

    NUp = WNew' * W;
    W .= WNew;
    X .= XNew;

    ################## S-step ##################
    S .= MUp*S*(NUp')

    S .= S .- alpha.*X'*df(X*S*W')*W;
    push!(DLRHistory, f(X*S*W'))
    println("residual: ",f(X*S*W'))
end

fig, ax = subplots()
ax[:plot](collect(1:length(sdHistory)),sdHistory, "k-", linewidth=2, label="sd", alpha=0.6)
ax[:plot](collect(1:length(DLRHistory)),DLRHistory, "r--", linewidth=2, label="DLR", alpha=0.6)
ax[:legend](loc="upper right")
ax.tick_params("both",labelsize=20) 
show()