from m2_utilities.flops import compute_flops

# Calculate FLOPS for a single step
flops = compute_flops(128, 4, backpropagate=True)
print(f"{flops:.6e}")

flops = compute_flops(512, 4, backpropagate=True)
print(f"{flops:.6e}")

flops = compute_flops(768, 4, backpropagate=True)
print(f"{flops:.6e}")