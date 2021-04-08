using CSV
using DataFrames
using PyPlot
using LinearAlgebra

function MatToVec(u::Array{Float64,2})
    N1 = size(u,1);
    N2 = size(u,2);
    
    y = zeros(N1*N2);
    for i = 1:N1
        for j = 1:N2
            #y[j+N2*(i-1)] = u[i,j];
            y[i+N1*(j-1)] = u[i,j];
        end
    end

    return y
end

function VecToMat(y::Array{Float64,1},N1,N2)
    
    u = zeros(N1,N2);
    for i = 1:N1
        for j = 1:N2
            #u[i,j] = y[j+N2*(i-1)];
            u[i,j] = y[i+N1*(j-1)];
        end
    end

    return u
end

X = CSV.read("surface_adjoint.csv",DataFrame)

X = CSV.File("surface_adjoint_clean.csv") |> Tables.matrix

grid = X[25:174,1];
df = X[25:174,10];
M = length(df);

N1 = 15#15;
N2 = 10#10;
r = 6;
u = VecToMat(df,N1,N2)

# compute low-rank representation
X,S,W = svd(u); 
    
# rank-r truncation:
X = X[:,1:r]; 
W = W[:,1:r];
S = Diagonal(S);
S = S[1:r, 1:r]; 

# plot 
fig, ax = subplots()
ax[:plot](grid,MatToVec(u), "k-", linewidth=2, label="df", alpha=0.6)
ax[:plot](grid,MatToVec(X*S*W'), "r--", linewidth=2, label="low-rank", alpha=0.6)
ax[:legend](loc="upper left")
ax.set_xlim([grid[1],grid[end]])
ax.set_xlabel("index x", fontsize=20);
show()
