import faiss
import torch
import time

# soft labels
# weighted convolution against kNN in feature space
# use FAISS for fast neighbor search on GPU

# Dictionary to hold the intermediate activation (embedding) captured by the hook
activation = {}

def get_activation(name):
    # This hook is called right before the forward pass of the hooked module (nn.Linear)
    def hook(model, input):
        # 'input' is a tuple, typically (data_tensor,)
        # input[0] is the input to the final layer, which is the embedding
        activation[name] = input[0].detach()
    return hook

@torch.no_grad()
def extract_embeddings(model_ori, loader, device, num_classes):
    model_ori.eval()
    all_embeddings = []
    all_labels = []
    hook_handle = None
    final_layer = None
    final_layer_name = None

    # --- Programmatic Embedding Extraction Setup ---
    
    # 1. Iterate over named modules in reverse to find the FINAL classification layer
    # This assumes the classification head is the last nn.Linear layer mapping to num_classes.
    for name, module in reversed(list(model_ori.named_modules())):
        if isinstance(module, torch.nn.Linear) and module.out_features == num_classes:
            final_layer = module
            final_layer_name = name
            break # Found the last classification head
    
    if final_layer is None:
        raise AttributeError(
            f"Could not find a clear final classification layer (nn.Linear with {num_classes} out_features). "
            "Model structure is too complex or non-standard for automatic embedding extraction."
        )

    # 2. Register the hook on the final layer to capture its input (the embedding)
    # register_forward_pre_hook captures the tensor IMMEDIATELY before the layer executes.
    hook_handle = final_layer.register_forward_pre_hook(get_activation(final_layer_name))
    print(f"\n[FAISS] Registered hook on final layer: {final_layer_name} to capture embedding.")
    
    print("[FAISS] Collecting all training embeddings...")
    start_time = time.time()
    
    # Iterate over the DataLoader, ensuring minimal memory footprint per batch
    for data, labels in loader:
        data = data.to(device)
        
        # Run the full forward pass to trigger the hook. We discard the logits output.
        _ = model_ori(data)
        
        # --- EXTRACT EMBEDDING FROM HOOK ---
        if final_layer_name not in activation:
             raise RuntimeError("Hook failed to capture activation. Check model structure and hook registration.")
        
        embeddings = activation[final_layer_name] # Already detached in the hook
        
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.detach().cpu())

        # Clear the captured activation dictionary for the next batch
        del activation[final_layer_name]

    # 3. Unregister the hook after collecting all data
    hook_handle.remove()
    print("[FAISS] Hook unregistered.")

    all_embeddings = torch.cat(all_embeddings, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)
    
    end_time = time.time()
    print(f"[FAISS] Collection complete: {all_embeddings.shape[0]} samples in {end_time - start_time:.2f}s.")
    return all_embeddings, all_labels # tuple of tensors

def build_faiss_index(embedding):
    # embeddings: torch.Tensor [N, D] on CPU or GPU
    emb = embedding.cpu().numpy().astype('float32')

    index = faiss.IndexFlatL2(emb.shape[1])        # exact L2
    gpu_index = faiss.index_cpu_to_all_gpus(index) # move to GPU(s)
    gpu_index.add(emb)
    return gpu_index

def query_index(index, query_vectors, k=10):
    q = query_vectors.cpu().numpy().astype('float32')
    distances, neighbors = index.search(q, k)
    return distances, neighbors # np arrays

@torch.no_grad()
def smooth_labels_uniform(
    labels, 
    device, 
    num_classes: int = 10,
    eps: float = 0.0, 
    c: float = 0.5,
):
    one_hot_labels = torch.nn.functional.one_hot(labels).to(device)
    uniform = (1.0 / num_classes) * torch.ones(*one_hot_labels.shape)
    uniform = uniform.to(device)
    smooth_labels = (1 - c * eps) * one_hot_labels + c * eps * uniform
    print(smooth_labels[:10])
    return smooth_labels

def smooth_labels_knn(
    embeddings, 
    labels,
    device,
    k: int = 5,
    sigma: float = 0.5,
    num_classes: int = 10,
) -> torch.Tensor:
    
    faiss_index = build_faiss_index(embeddings)

    distances_sq, neighbor_indices = query_index(faiss_index, embeddings, k)
    distances_sq = torch.from_numpy(distances_sq).to(device)
    neighbor_indices = torch.from_numpy(neighbor_indices).to(device)
    
    N, D = embeddings.shape
    
    weights = torch.exp(-distances_sq / (2 * sigma**2)) # (N, K)
    
    # 3.2: Gather the corresponding labels for the neighbors
    labels = labels.to(device).long()
    neighbor_labels = labels[neighbor_indices] # (N, K)

    # 3.3: One-Hot Encode the labels for the "convolution"
    one_hot_labels = torch.zeros(
        (N, k, num_classes), 
        dtype=torch.float32, 
        device=device # Use the device of the embeddings
    )
    # Scatter the value 1 into the correct class index
    one_hot_labels.scatter_(2, neighbor_labels.unsqueeze(-1), 1)

    # 3.4: Apply the Weighted Sum (The Convolution/Smoothing)
    weighted_votes = weights.unsqueeze(-1) * one_hot_labels # (N, K, num_classes)
    
    # Sum over the K neighbors dimension (dimension 1)
    smoothed_output = torch.sum(weighted_votes, dim=1) # (N, num_classes)
    
    # 3.5: Normalize the smoothed output to get a proper probability distribution
    # Add a small epsilon to the denominator to prevent division by zero
    smoothed_output_normalized = smoothed_output / (smoothed_output.sum(dim=1, keepdim=True) + 1e-6)

    ''' PRINT '''
    print("Distance_sq:", distances_sq[:10])
    print("Neighbor labels:", neighbor_labels[:10])
    print("Smoothed labels:", smoothed_output_normalized[:10])

    orig = labels.unsqueeze(1).expand_as(neighbor_labels)     # (N, K)

    # A sample is "nontrivially soft" if at least one neighbor has a different class
    has_distinct_neighbor = (neighbor_labels != orig).any(dim=1)   # (N,)
    proportion_soft = has_distinct_neighbor.float().mean().item()

    print(f"Proportion of nontrivially-soft labels: {proportion_soft:.4f}")
    
    return smoothed_output_normalized.detach().to(device).requires_grad_(False)
