using LinearAlgebra
using PyPlot

function f1(x::Array{Float64,2},y::Array{Float64,1})
    return 0.5*norm(x - y*y',2).^2;
end

function df1(x::Array{Float64,2},y::Array{Float64,1})
    return x-y*y';
end

function f2(x::Array{Float64,2},y::Array{Float64,1},A::Array{Float64,2})
    return 0.5*norm((x - y*y').*A,2).^2;
end

function df2(x::Array{Float64,2},y::Array{Float64,1},A::Array{Float64,2})
    return (x-y*y').*A.*A;
end

function DLRRankAdaptive(f::Function,df::Function,r::Int=5,rMaxTotal::Int=20,epsAdapt::Float64=1e-1)
x = zeros(N,N);
# Low-rank approx of init data:
X,S,W = svd(x); 
    
# rank-r truncation:
X = X[:,1:r]; 
W = W[:,1:r];
S = Diagonal(S);
S = Array(S[1:r, 1:r]); 

K = zeros(N,r);
L = zeros(N,r);

DLRadaptHistory = [f(X*S*W')];

    while f(X*S*W') > eps
        
        gradient = df(X*S*W');

        ###### K-step ######
        K = X*S;
        K1 = K .- alpha*gradient*W;

        K1 = [K1 X];

        XNew,STmp = qr(K1);
        XNew = XNew[:,1:2*r];

        MUp = XNew' * X;
        ###### L-step ######
        L = W*S';

        L1 = L .- alpha*(X'*gradient)';

        L1 = [L1 W];
        WNew,STmp = qr(L1);
    
        WNew = WNew[:,1:2*r];

        NUp = WNew' * W;

        ################## S-step ##################
        X = XNew;
        W = WNew;
        STilde = MUp*S*(NUp');

        S = STilde .- alpha.*X'*df(X*STilde*W')*W;

        ################## truncate ##################

        # Compute singular values of S1 and decide how to truncate:
        U,D,V = svd(S);
        rmax = -1;
        S .= zeros(size(S));


        tmp = 0.0;
        tol = epsAdapt*norm(D);
        
        rmax = Int(floor(size(D,1)/2));
        
        for j=1:2*rmax
            tmp = sqrt(sum(D[j:2*rmax]).^2);
            if(tmp<tol)
                rmax = j;
                break;
            end
        end
        
        rmax = min(rmax,rMaxTotal);
        rmax = max(rmax,2);

        for l = 1:rmax
            S[l,l] = D[l];
        end

        # if 2*r was actually not enough move to highest possible rank
        if rmax == -1
            rmax = rMaxTotal;
        end

        # update solution with new rank
        XNew = XNew*U;
        WNew = WNew*V;

        # update solution with new rank
        S = S[1:rmax,1:rmax];
        X = XNew[:,1:rmax];
        W = WNew[:,1:rmax];

        # update rank
        r = rmax;

        push!(DLRadaptHistory, f(X*S*W'))
        println("residual DLR: ",f(X*S*W'), "  -> r = ",r)
    
    end
    return DLRadaptHistory
end

N = 40;
x = zeros(N,N);
y = rand(N);
A = rand(N,N)

#f = x-> f1(x,y);
#df = x-> df1(x,y);

f = x-> f2(x,y,A);
df = x-> df2(x,y,A);

alpha = 5e-1;
eps = 1e-5;

sdHistory = [f(x)];

# steepest descent
while f(x) > eps
    x .= x - alpha*df(x);
    push!(sdHistory, f(x))
    println(f(x))
end

############# DLR #############
r = 5;
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
    L .= W*S';

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

DLRadaptHistory = DLRRankAdaptive(f,df)

fig, ax = subplots()
ax[:plot](collect(1:length(sdHistory)),sdHistory, "k-", linewidth=2, label="sd", alpha=0.6)
ax[:plot](collect(1:length(DLRHistory)),DLRHistory, "r--", linewidth=2, label="DLR", alpha=0.6)
ax[:plot](collect(1:length(DLRadaptHistory)),DLRadaptHistory, "g:", linewidth=2, label="DLR adapt", alpha=0.6)
ax[:legend](loc="upper right")
ax.set_yscale("log")
ax.tick_params("both",labelsize=20) 
show()