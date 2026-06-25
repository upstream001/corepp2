\begin{table*}[t]
\centering
\caption{Detailed architecture of the proposed Point-MAE-style point cloud encoder.
$d_z$ denotes the latent vector dimension, which is set to 32 in our experiments.
BN: BatchNorm; LN: LayerNorm; FPS: Farthest Point Sampling; KNN: $k$-Nearest Neighbors;
MHSA: Multi-Head Self-Attention.}
\label{tab:encoder_arch}
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{llccl}
\toprule
\textbf{Stage} & \textbf{Module} & \textbf{Key Params} & \textbf{Channels} & \textbf{Output Size} \\
\midrule
Input
& Point cloud
& $N=2048$
& 3
& $2048$ \\
\midrule
Patch Grouping
& FPS sampling
& $G=64$
& 3
& $64$ \\

& KNN grouping
& $k=32$
& 3
& $64 \times 32$ \\

& Local normalization
& subtract patch center
& 3
& $64 \times 32$ \\
\midrule
Patch Embedding
& Shared MLP
& Conv2d + BN + GELU
& $3 \to 128 \to 128 \to 384$
& $64 \times 32$ \\

& Intra-patch aggregation
& max pooling
& $384$
& $64$ \\
\midrule
Position Encoding
& MLP on patch centers
& Linear + GELU + Linear
& $3 \to 384 \to 384$
& $64$ \\
\midrule
Token Assembly
& Token + position embedding
& element-wise addition
& $384$
& $64$ \\
\midrule
Transformer Encoder
& Transformer block $\times 8$
& MHSA, $h=8$, MLP ratio$=4$
& $384 \to 384$
& $64$ \\

& Final normalization
& LayerNorm
& $384$
& $64$ \\
\midrule
Global Aggregation
& Attention pooling
& score MLP + softmax
& $384 \to 192 \to 1$
& $1$ \\

& Max Pool + Avg Pool
& over patch tokens
& $384 + 384 \to 768$
& $1$ \\

& Concatenation
& attn + max + avg
& $1152$
& $1$ \\
\midrule
Head
& Fully connected MLP
& LN + GELU + Dropout
& $1152 \to 512 \to 256 \to d_z$
& $1$ \\
\bottomrule
\end{tabular}
\end{table*}
