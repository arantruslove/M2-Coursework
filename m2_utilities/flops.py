def matmul(m, n, p):
    """Compute the number of FLOPS in a single matrix multiplication."""
    return m * p * (2 * n - 1)


def dot_product(n):
    """Compute the number of FLOPS in a dot product of two vectors."""
    # Dot product is a special case of a matrix multiplication
    return matmul(1, n, 1)


def embedding(n_tokens, d_model):
    """Compute the number of FLOPS in the embedding layers."""
    return n_tokens * d_model


def attention_value(n_tokens, d_h, d_model):
    """Compute the number of FLOPS for determining a query, key, or value vector."""
    single_value_flops = matmul(d_h, d_model, 1) + d_h
    return n_tokens * single_value_flops


def softmax(n_tokens, vector_size):
    """Compute the number of FLOPS for applying Softmax to a vector."""
    expon_flops = vector_size * 10
    sum_flops = vector_size - 1
    division_flops = vector_size

    return n_tokens * (expon_flops + sum_flops + division_flops)


def weighted_sum_values(n_tokens, d_h):
    """
    Compute the number of FLOPS for calculating the weights average of values associated
    with a single token.
    """
    # Product of each vector with its attention coefficient
    product_flops = n_tokens * d_h

    # Summing the resulting vectors
    sum_flops = (n_tokens - 1) * d_h

    return product_flops + sum_flops


def multi_head_self_attention(n_tokens, n_heads, d_model):
    """Compute the number of FLOPS in a single multi-head self-attention layer."""
    # Head size (query/key/value size)
    d_h = d_model / n_heads

    # Compute queries, keys, and values
    attention_value_flops = 3 * attention_value(n_tokens, d_h, d_model)

    # Query-key dot products
    qk_dot_flops = dot_product(d_h) * n_tokens**2

    # Divide attention value by sqrt of head size
    sqrt_flops = 10
    divide_flops = n_tokens**2

    # Overall Softmax operation
    softmax_flops = sqrt_flops + divide_flops + softmax(n_tokens, n_tokens)

    # Compute output values after self-attention is applied
    ouput_vals_flops = n_tokens * weighted_sum_values(n_tokens, d_h)

    # Final linear projection of concatenated heads
    linear_projection_flops = n_tokens * matmul(d_model, d_model, 1)

    # Multiplying by the number of heads
    total_flops = (
        n_heads
        * (attention_value_flops + qk_dot_flops + softmax_flops + ouput_vals_flops)
        + linear_projection_flops
    )
    return total_flops


def rms_norm(n_tokens, d_model):
    """Compute the FLOPS for RMS normalisation."""
    # Compute the RMS coefficient of the vector
    square_flops = d_model * 10
    sum_flops = d_model - 1
    divide_flops = 1
    root_flops = 10

    rms_flops = root_flops + divide_flops + sum_flops + square_flops

    # Dividing each feature by RMS coefficient
    divide_features_flops = d_model

    # Element-wise multiplication with gamma vector
    multiply_flops = d_model

    # Multiply by the number of input tokens
    total_flops = n_tokens * (rms_flops + divide_features_flops + multiply_flops)
    return total_flops


def add_residual(n_tokens, d_model):
    """Compute FLOPS for adding back the residual."""
    return n_tokens * d_model


def swish():
    """Compute FLOPS for Swish activation on a single input."""
    denominator_flops = 1 + 1 + 10
    divide_flops = 1
    return denominator_flops + divide_flops


def ffn(n_tokens, d_model, hidden_size):
    """
    Compute FLOPS for the feed-forwards network applied to all input tokens. FFN uses
    SwiGLU activation and contains no biases in Qwen2.5.
    """
    # FFN applied to a single token
    w1_flops = matmul(hidden_size, d_model, 1)
    swish_flops = hidden_size * swish()
    w2_flops = matmul(hidden_size, d_model, 1)
    element_wise_flops = hidden_size
    w3_flops = matmul(d_model, hidden_size, 1)

    # Multiply by number of inputs
    total_flops = n_tokens * (
        w1_flops + swish_flops + w2_flops + element_wise_flops + w3_flops
    )
    return total_flops


def final_linear(n_tokens, d_model, vocab_size):
    """Compute FLOPS for the final linear transformation."""
    return matmul(vocab_size, d_model, n_tokens) + n_tokens * vocab_size


def block(n_tokens, n_heads, d_model, hidden_size):
    """Compute FLOPS for a single self-attention block."""
    # Multi-head self-attention
    total_flops = rms_norm(n_tokens, d_model)
    total_flops += multi_head_self_attention(n_tokens, n_heads, d_model)
    total_flops += add_residual(n_tokens, d_model)

    # Feed-forward network
    total_flops += rms_norm(n_tokens, d_model)
    total_flops += ffn(n_tokens, d_model, hidden_size)
    total_flops += add_residual(n_tokens, d_model)
    return total_flops


def compute_flops(
    n_tokens,
    batch_size=1,
    n_layers=24,
    n_heads=14,
    vocab_size=151936,
    d_model=896,
    hidden_size=4864,
    inference=False,
    backpropagate=False,
):
    """Compute FLOPS for a forward pass of the network."""

    # Adding positional embeddings
    total_flops = embedding(n_tokens, d_model)

    # Self-attention blocks
    for _ in range(n_layers):
        total_flops += block(n_tokens, n_heads, d_model, hidden_size)

    # Linear transformation and softmax
    post_blocks_flops = rms_norm(n_tokens, d_model)
    post_blocks_flops += final_linear(n_tokens, d_model, vocab_size)
    post_blocks_flops += softmax(n_tokens, vocab_size)

    # Only applied to a single token during inference
    if inference:
        post_blocks_flops /= n_tokens
    total_flops += post_blocks_flops

    # Multiply by batch size
    total_flops *= batch_size

    # Forward + backprop
    if backpropagate:
        return 3 * total_flops

    # Only forward
    return total_flops


def compute_flops_gen(n_context, n_generate, **kwargs):
    """
    Compute the number of flops when generating 'n_generate' tokens with an input size
    of 'n_input'.
    """
    total_flops = 0
    for i in range(n_generate):
        total_flops += compute_flops(n_context + i, inference=True, **kwargs)
    return total_flops
