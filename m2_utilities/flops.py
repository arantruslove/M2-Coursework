def matmul(m, n, p):
    """Compute the number of floating-point operations (FLOPS) for matrix multiplication.

    Args:
        m (int): Number of rows in the first matrix.
        n (int): Number of columns in the first matrix (and rows in the second matrix).
        p (int): Number of columns in the second matrix.

    Returns:
        int: Total number of FLOPS required for the matrix multiplication.
    """
    return m * p * (2 * n - 1)


def dot_product(n):
    """Compute the number of FLOPS for a dot product of two vectors.

    Args:
        n (int): Length of the vectors.

    Returns:
        int: Total FLOPS for the dot product.
    """
    return matmul(1, n, 1)


def embedding(n_tokens, hidden_size):
    """Compute the number of FLOPS in an embedding layer.

    Args:
        n_tokens (int): Number of input tokens.
        hidden_size (int): Dimension of the model embeddings (hidden size).

    Returns:
        int: Total FLOPS for the embedding layer.
    """
    return n_tokens * hidden_size


def attention_value(n_tokens, d_h, hidden_size):
    """Compute FLOPS for computing query, key, or value vectors in attention.

    Args:
        n_tokens (int): Number of tokens.
        d_h (int): Dimension per attention head (head size).
        hidden_size (int): Hidden size.

    Returns:
        float: Total FLOPS for attention value computation.
    """
    single_value_flops = matmul(d_h, hidden_size, 1) + d_h
    return n_tokens * single_value_flops


def softmax(n_tokens, vector_size):
    """Compute FLOPS for the softmax operation.

    Args:
        n_tokens (int): Number of tokens.
        vector_size (int): Size of the vector for which softmax is computed.

    Returns:
        int: Total FLOPS for the softmax operation.
    """
    expon_flops = vector_size * 10
    sum_flops = vector_size - 1
    division_flops = vector_size

    return n_tokens * (expon_flops + sum_flops + division_flops)


def weighted_sum_values(n_tokens, d_h):
    """Compute FLOPS for the weighted sum of value vectors in attention.

    Args:
        n_tokens (int): Number of tokens.
        d_h (int): Head size.

    Returns:
        int: Total FLOPS for computing weighted sum.
    """
    product_flops = n_tokens * d_h
    sum_flops = (n_tokens - 1) * d_h

    return product_flops + sum_flops


def multi_head_self_attention(n_tokens, n_heads, hidden_size):
    """Compute FLOPS for a multi-head self-attention layer.

    Args:
        n_tokens (int): Number of tokens.
        n_heads (int): Number of attention heads.
        hidden_size (int): Hidden size.

    Returns:
        float: Total FLOPS for the attention layer.
    """
    d_h = hidden_size / n_heads

    attention_value_flops = 3 * attention_value(n_tokens, d_h, hidden_size)
    qk_dot_flops = dot_product(d_h) * n_tokens**2
    sqrt_flops = 10
    divide_flops = n_tokens**2
    softmax_flops = sqrt_flops + divide_flops + softmax(n_tokens, n_tokens)
    ouput_vals_flops = n_tokens * weighted_sum_values(n_tokens, d_h)
    linear_projection_flops = n_tokens * matmul(hidden_size, hidden_size, 1)

    total_flops = (
        n_heads
        * (attention_value_flops + qk_dot_flops + softmax_flops + ouput_vals_flops)
        + linear_projection_flops
    )
    return total_flops


def rms_norm(n_tokens, hidden_size):
    """Compute FLOPS for RMS normalization.

    Args:
        n_tokens (int): Number of tokens.
        hidden_size (int): Hidden size.

    Returns:
        int: Total FLOPS for RMSNorm.
    """
    square_flops = hidden_size * 10
    sum_flops = hidden_size - 1
    divide_flops = 1
    root_flops = 10

    rms_flops = root_flops + divide_flops + sum_flops + square_flops
    divide_features_flops = hidden_size
    multiply_flops = hidden_size

    return n_tokens * (rms_flops + divide_features_flops + multiply_flops)


def add_residual(n_tokens, hidden_size):
    """Compute FLOPS for residual addition.

    Args:
        n_tokens (int): Number of tokens.
        hidden_size (int): Hidden size.

    Returns:
        int: Total FLOPS for adding residuals.
    """
    return n_tokens * hidden_size


def swish():
    """Compute FLOPS for Swish activation function.

    Returns:
        int: Total FLOPS for a single Swish operation.
    """
    denominator_flops = 1 + 1 + 10
    divide_flops = 1
    return denominator_flops + divide_flops


def ffn(n_tokens, hidden_size, intermediate_size):
    """Compute FLOPS for the feed-forward network using SwiGLU.

    Args:
        n_tokens (int): Number of tokens.
        hidden_size (int): Hidden size.
        intermediate_size (int): Size of the hidden layer of the FFN (intermediate size).

    Returns:
        int: Total FLOPS for the feed-forward layer.
    """
    w1_flops = matmul(intermediate_size, hidden_size, 1)
    swish_flops = intermediate_size * swish()
    w2_flops = matmul(intermediate_size, hidden_size, 1)
    element_wise_flops = intermediate_size
    w3_flops = matmul(hidden_size, intermediate_size, 1)

    return n_tokens * (
        w1_flops + swish_flops + w2_flops + element_wise_flops + w3_flops
    )


def final_linear(n_tokens, hidden_size, vocab_size):
    """Compute FLOPS for the final projection to vocabulary logits.

    Args:
        n_tokens (int): Number of tokens.
        hidden_size (int): Hidden size.
        vocab_size (int): Size of the output vocabulary.

    Returns:
        int: Total FLOPS for final linear projection.
    """
    return matmul(vocab_size, hidden_size, n_tokens) + n_tokens * vocab_size


def block(n_tokens, n_heads, hidden_size, intermediate_size):
    """Compute FLOPS for a single Transformer block.

    Args:
        n_tokens (int): Number of tokens.
        n_heads (int): Number of attention heads.
        hidden_size (int): Hidden size.
        intermediate_size (int): Hidden size of the FFN layer.

    Returns:
        float: Total FLOPS for one block.
    """
    total_flops = rms_norm(n_tokens, hidden_size)
    total_flops += multi_head_self_attention(n_tokens, n_heads, hidden_size)
    total_flops += add_residual(n_tokens, hidden_size)
    total_flops += rms_norm(n_tokens, hidden_size)
    total_flops += ffn(n_tokens, hidden_size, intermediate_size)
    total_flops += add_residual(n_tokens, hidden_size)
    return total_flops


def compute_flops(
    n_tokens,
    batch_size=1,
    n_layers=24,
    n_heads=14,
    vocab_size=151936,
    hidden_size=896,
    intermediate_size=4864,
    inference=False,
    backpropagate=False,
):
    """Compute FLOPS for a complete forward pass (and optionally backpropagation) of the model.

    Args:
        n_tokens (int): Number of input tokens.
        batch_size (int, optional): Batch size. Defaults to 1.
        n_layers (int, optional): Number of Transformer layers. Defaults to 24.
        n_heads (int, optional): Number of attention heads. Defaults to 14.
        vocab_size (int, optional): Vocabulary size. Defaults to 151936.
        hidden_size (int, optional): Hidden size. Defaults to 896.
        intermediate_size (int, optional): FFN intermediate size. Defaults to 4864.
        inference (bool, optional): Whether in inference mode. Defaults to False.
        backpropagate (bool, optional): Whether to include backprop FLOPS. Defaults to False.

    Returns:
        float: Total FLOPS for the pass.
    """
    total_flops = embedding(n_tokens, hidden_size)

    for _ in range(n_layers):
        total_flops += block(n_tokens, n_heads, hidden_size, intermediate_size)

    post_blocks_flops = rms_norm(n_tokens, hidden_size)
    post_blocks_flops += final_linear(n_tokens, hidden_size, vocab_size)
    post_blocks_flops += softmax(n_tokens, vocab_size)

    if inference:
        post_blocks_flops /= n_tokens

    total_flops += post_blocks_flops
    total_flops *= batch_size

    if backpropagate:
        return 3 * total_flops

    return total_flops


def compute_flops_gen(n_context, n_generate, **kwargs):
    """Compute FLOPS for generating tokens autoregressively.

    Args:
        n_context (int): Context length.
        n_generate (int): Number of tokens to generate.
        **kwargs: Additional arguments for `compute_flops`.

    Returns:
        float: Total FLOPS for generating all tokens.
    """
    total_flops = 0
    for i in range(n_generate):
        total_flops += compute_flops(n_context + i, inference=True, **kwargs)
    return total_flops
