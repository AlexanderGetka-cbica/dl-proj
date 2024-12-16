from conv_CBilA import BilinearCNN, Bilinear
from einops import *
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decompose_bilinear_layer(embedding_layer, bilinear_layer, unembedding_layer):
    """The function to decompose a single-layer model into eigenvalues and eigenvectors."""
    # Show iter and handle layer type issues
    #print(f"Iteration {iter}")
    if not isinstance(bilinear_layer, Bilinear):
        print("Passed bilinear layer is not bilinear, stopping decomposition.")
        return None, None
    if not isinstance(unembedding_layer, torch.nn.Linear):
        print("Passed unembedding layer is not a linear layer, stopping decomposition.")
        return None, None
    if not isinstance(embedding_layer, torch.nn.Linear):
        print("Passed embedding layer is not a linear layer, stopping decomposition.")
        return None, None
    
    # Split the bilinear layer into the left and right components
    w_l = bilinear_layer.w_l
    w_r = bilinear_layer.w_r    
    w_u = unembedding_layer.weight
    # Consider the embedding weights to be always identity
    w_e = embedding_layer.weight

    #print(w_l)
    #print(w_r)
    #print(w_u)
    #print(w_e)
    # Compute the third-order (bilinear) tensor
    b = einsum(w_u, w_l, w_r, "cls out, out in1, out in2 -> cls in1 in2")
    
    # Symmetrize the tensor
    assert b.shape[1] == b.shape[2], "Symmetrization requires square tensors"   
    b = 0.5 * (b + b.mT)

    # Perform the eigendecomposition
    vals, vecs = torch.linalg.eigh(b)
    
    # Project the eigenvectors back to the input space
    vecs = einsum(vecs, w_e, "cls emb comp, emb inp -> cls comp inp")
    
    # Return the eigenvalues and eigenvectors
    return vals, vecs


# Load BilinearCNN from file
#model = BilinearCNN(base_channels=64, num_classes=10, bias=False, gate='none')
#model.load_state_dict("trained_bilinear_model.pth")
#model = model.to(device)
#model.eval()

# We can iterate back through bilinear layers.
#eigs = {}
#eigenvectors, eigenvalues = decompose_bilinear_layer(model.bilinear, model.fc)
#eigs[0] = (eigenvalues, eigenvectors)
# this first result has eigenvectors in the space of feature maps.










