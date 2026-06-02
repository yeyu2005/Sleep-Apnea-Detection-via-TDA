"""TDA pipeline: Takens embedding, Vietoris-Rips, Persistence Image using giotto-tda."""
import numpy as np

try:
    from gtda.time_series import TakensEmbedding
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceImage
except Exception:
    TakensEmbedding = None
    VietorisRipsPersistence = None
    PersistenceImage = None



def compute_persistence_image(rr_window, tau, dim, grid_size=32, sigma=0.1, return_diagram=False):
    if TakensEmbedding is None or VietorisRipsPersistence is None or PersistenceImage is None:
        raise RuntimeError('giotto-tda is required for TDA pipeline')
    
    rr = np.asarray(rr_window, dtype=float)
    
    # FIX 1: Strip out any NaN or Inf values that survived interpolation
    rr = rr[np.isfinite(rr)]
    
    # FIX 2: Check if the stripped array is long enough for the chosen Takens parameters.
    # Takens embedding requires at least (dim - 1) * tau + 1 points.
    min_required_points = (dim - 1) * tau + 1
    
    if len(rr) < min_required_points:
        # If the window is too corrupted/short, return an empty topological space (all zeros)
        # This prevents crashes and safely represents a "lack of structure" to the neural net.
        empty_pi = np.zeros(grid_size * grid_size)
        if return_diagram:
            return empty_pi, np.empty((0, 3))
        return empty_pi

    te = TakensEmbedding(time_delay=tau, dimension=dim)
    X_emb = te.fit_transform(rr.reshape(1, -1))  # (n_samples, n_timestamps, dim)
    
    vr = VietorisRipsPersistence(homology_dimensions=[0, 1])
    diagrams = vr.fit_transform(X_emb.reshape(X_emb.shape[0], -1, X_emb.shape[-1]))
    
    pi = PersistenceImage(sigma=sigma, n_bins=grid_size)
    pis = pi.fit_transform(diagrams)
    
    if return_diagram:
        return pis.reshape(pis.shape[0], -1)[0], diagrams[0]
    return pis.reshape(pis.shape[0], -1)[0]


def cache_persistence_images(rr_dir, out_dir, tau, dim, grid_size=32, sigma=0.1, pattern='*.npy', save_diagrams=False):
    """Compute and cache persistence images for all RR window files in `rr_dir`."""
    import glob
    import os
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(rr_dir, pattern)))
    if TakensEmbedding is None:
        raise RuntimeError('giotto-tda is required for caching persistence images')
    
    success_count = 0
    for f in files:
        try:
            rr = np.load(f)
        except Exception:
            continue
            
        try:
            if save_diagrams:
                pi, dgm = compute_persistence_image(rr, tau, dim, grid_size=grid_size, sigma=sigma, return_diagram=True)
            else:
                pi = compute_persistence_image(rr, tau, dim, grid_size=grid_size, sigma=sigma)
        except Exception as e:
            # FIX 3: Warn instead of crashing the entire loop if an unexpected math error occurs
            print(f"Warning: TDA failed for {os.path.basename(f)} - {e}. Saving zero array.")
            pi = np.zeros(grid_size * grid_size)
            if save_diagrams:
                dgm = np.empty((0, 3))
                
        outpath = os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + '_pi.npy')
        np.save(outpath, pi)
        
        if save_diagrams:
            outd = os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + '_dgm.npy')
            try:
                np.save(outd, dgm)
            except Exception:
                pass
        success_count += 1
        
    print(f"Successfully processed {success_count} files.")
    return True